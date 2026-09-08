#!/usr/bin/env python3
"""Validate continuous OAK-4P camera/IMU datasets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dohc2_c4lib.aprilgrid_common import CAMERA_IDS, SENSOR_RESOLUTION, assess_image, create_detector, write_json_atomic
from dohc2_c4lib.oak4p_dataset import (
    DEFAULT_MAX_PIXEL_PHASE_SPREAD,
    load_camera_rows,
    load_imu_rows,
    percentile,
    pixel_phase_metrics,
    read_csv_rows,
)


MAX_APRILGRID_DETECTION_SAMPLES_PER_CAMERA = 200


def read_stored_image(image_path: Path, metadata: dict) -> np.ndarray | None:
    """Decode a recorded image, including headerless GRAY8 and NV12 frames."""
    image_path = Path(image_path)
    suffix = image_path.suffix.lower()
    if suffix not in {".gray8", ".nv12"}:
        return cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    width, height = metadata.get("sensor_resolution", SENSOR_RESOLUTION)
    pixels = np.fromfile(image_path, dtype=np.uint8)
    width, height = int(width), int(height)
    expected = width * height if suffix == ".gray8" else width * height * 3 // 2
    if pixels.size != expected:
        return None
    if suffix == ".gray8":
        return pixels.reshape((height, width))
    # OAK's full-resolution NV12 output is an 8-bit Y plane followed by an
    # interleaved UV plane.  Convert only for validation; the recorder keeps
    # the original bytes unchanged on disk.
    return cv2.cvtColor(pixels.reshape((height * 3 // 2, width)), cv2.COLOR_YUV2BGR_NV12)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="检查OAK-4P连续四目/IMU数据")
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--strict-sync-us", type=int)
    parser.add_argument("--min-groups", type=int, default=2)
    parser.add_argument("--min-aprilgrid-bins", type=int, default=6)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--skip-image-decode", action="store_true")
    args = parser.parse_args(argv)
    if args.min_groups < 1 or not 1 <= args.min_aprilgrid_bins <= 9:
        parser.error("min-groups必须>=1，min-aprilgrid-bins必须为1..9")
    return args


def rate_and_gaps(timestamps: list[int], expected_hz: float) -> dict:
    deltas = np.diff(np.asarray(timestamps, dtype=np.int64)) if len(timestamps) > 1 else np.asarray([])
    duration_s = (timestamps[-1] - timestamps[0]) / 1e9 if len(timestamps) > 1 else 0.0
    actual_hz = (len(timestamps) - 1) / duration_s if duration_s > 0 else 0.0
    expected_period_ns = 1e9 / expected_hz if expected_hz > 0 else math.inf
    gaps = int(np.count_nonzero(deltas > expected_period_ns * 1.8)) if deltas.size else 0
    return {
        "count": len(timestamps),
        "duration_s": duration_s,
        "actual_hz": actual_hz,
        "gap_count": gaps,
        "max_gap_ms": float(np.max(deltas) / 1e6) if deltas.size else None,
    }


def coverage_bin(center, width, height):
    column = min(2, max(0, int(center[0] * 3 / width)))
    row = min(2, max(0, int(center[1] * 3 / height)))
    return row * 3 + column


def validate_dataset(
    session_dir: Path,
    strict_sync_us: int | None = None,
    min_groups: int = 2,
    min_aprilgrid_bins: int = 6,
    decode_images: bool = True,
    update_metadata: bool = True,
):
    session_dir = Path(session_dir)
    metadata_path = session_dir / "session.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"缺少文件：{metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    strict_sync_us = int(strict_sync_us or metadata.get("strict_sync_us", 1_000))
    expected_fps = float(metadata.get("sensor_fps", 30.0))
    errors, warnings = [], []
    camera_summary = {}
    camera_timestamp_sets = {}
    detector = (
        create_detector()
        if metadata.get("mode") in ("cam_calib", "imu_calib") and decode_images
        else None
    )

    for name in CAMERA_IDS:
        path = session_dir / name / "data.csv"
        if not path.is_file():
            errors.append(f"{name}缺少data.csv")
            continue
        try:
            rows = load_camera_rows(session_dir, name)
        except (KeyError, ValueError, OSError) as exc:
            errors.append(f"{name} data.csv无效：{exc}")
            continue
        timestamps = [row.timestamp_ns for row in rows]
        camera_timestamp_sets[name] = set(timestamps)
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            errors.append(f"{name}时间戳未严格递增")
        missing, corrupt, wrong_size = 0, 0, 0
        coverage = set()
        exposures = set()
        phase_spreads = []
        detection_stride = max(
            1,
            math.ceil(len(rows) / MAX_APRILGRID_DETECTION_SAMPLES_PER_CAMERA),
        )
        for row_index, row in enumerate(rows):
            image_path = session_dir / name / "data" / row.filename
            exposures.add(row.exposure_us)
            if not image_path.is_file():
                missing += 1
                continue
            if not decode_images:
                continue
            image = read_stored_image(image_path, metadata)
            if image is None:
                corrupt += 1
                continue
            if (image.shape[1], image.shape[0]) != SENSOR_RESOLUTION:
                wrong_size += 1
            gray = (
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if image.ndim == 3
                else image
            )
            if gray.ndim == 2 and gray.shape[0] >= 8 and gray.shape[1] >= 8:
                phase_spreads.append(float(pixel_phase_metrics(gray)["spread"]))
            if detector is not None and row_index % detection_stride == 0:
                _, assessment = assess_image(image, detector)
                if assessment.board_center is not None:
                    coverage.add(coverage_bin(assessment.board_center, image.shape[1], image.shape[0]))
        if missing or corrupt or wrong_size:
            errors.append(
                f"{name}图像异常：missing={missing}, corrupt={corrupt}, wrong_size={wrong_size}"
            )
        expected_sequence_step = int(metadata.get("skip_frames", 0)) + 1
        expected_recorded_fps = expected_fps / expected_sequence_step
        rate = rate_and_gaps(timestamps, expected_recorded_fps)
        sequence_gaps = sum(
            max(0, (right.sequence_num - left.sequence_num) // expected_sequence_step - 1)
            for left, right in zip(rows, rows[1:])
            if right.sequence_num > left.sequence_num
        )
        rate["sequence_gap_count"] = sequence_gaps
        if sequence_gaps:
            errors.append(f"{name}检测到{sequence_gaps}个未记录的设备帧")
        if len(rows) < 2:
            errors.append(f"{name}有效帧不足2")
        elif rate["actual_hz"] < expected_recorded_fps * 0.8:
            warnings.append(
                f"{name}实际落盘帧率{rate['actual_hz']:.2f}低于目标"
                f"{expected_recorded_fps:.2f}"
            )
        if detector is not None and len(coverage) < min_aprilgrid_bins:
            warnings.append(f"{name} AprilGrid仅覆盖{len(coverage)}个区域")
        phase_limit = float(
            metadata.get("max_pixel_phase_spread", DEFAULT_MAX_PIXEL_PHASE_SPREAD)
        )
        phase_p50 = percentile(phase_spreads, 50)
        phase_max = max(phase_spreads) if phase_spreads else None
        if phase_p50 is not None and phase_p50 > phase_limit:
            errors.append(
                f"{name} 2x2像素相位差中位数{phase_p50:.2f}>{phase_limit:.2f}，"
                "疑似未去马赛克Bayer数据"
            )
        camera_summary[name] = {
            **rate,
            "missing": missing,
            "corrupt": corrupt,
            "wrong_size": wrong_size,
            "exposure_values_us": sorted(exposures),
            "expected_recorded_fps": expected_recorded_fps,
            "aprilgrid_coverage_bins": sorted(coverage),
            "pixel_phase_spread_p50": phase_p50,
            "pixel_phase_spread_max": phase_max,
        }

    groups_path = session_dir / "groups.csv"
    groups = read_csv_rows(groups_path) if groups_path.is_file() else []
    if len(groups) < min_groups:
        errors.append(f"同步组{len(groups)}<{min_groups}")
    try:
        skews = [int(row["skew_us"]) for row in groups]
    except (KeyError, ValueError) as exc:
        errors.append(f"groups.csv无效：{exc}")
        skews = []
    previous_reference = None
    for index, group in enumerate(groups):
        try:
            reference = int(group["reference_timestamp_ns"])
            stamps = {
                name: int(group[f"{name}_timestamp_ns"]) for name in CAMERA_IDS
            }
        except (KeyError, ValueError) as exc:
            errors.append(f"同步组{index}字段无效：{exc}")
            continue
        if previous_reference is not None and reference <= previous_reference:
            errors.append("同步组参考时间戳未严格递增")
        previous_reference = reference
        missing = [
            name for name, stamp in stamps.items()
            if stamp not in camera_timestamp_sets.get(name, set())
        ]
        if missing:
            errors.append(f"同步组{index}引用不存在的相机帧：{missing}")
        measured_skew = math.ceil((max(stamps.values()) - min(stamps.values())) / 1_000)
        if measured_skew != int(group["skew_us"]):
            errors.append(
                f"同步组{index}偏差字段为{group['skew_us']}，重新计算为{measured_skew} us"
            )
    sync_summary = {
        "groups": len(groups),
        "p50_us": percentile(skews, 50),
        "p95_us": percentile(skews, 95),
        "p99_us": percentile(skews, 99),
        "max_us": max(skews) if skews else None,
        "strict_limit_us": strict_sync_us,
    }
    if skews and max(skews) > strict_sync_us:
        warnings.append(f"组内最大偏差{max(skews)}>{strict_sync_us} us，属于degraded数据")

    dropped = {name: int(value) for name, value in metadata.get("dropped", {}).items()}
    if any(dropped.values()):
        errors.append(f"采集期间发生队列丢包：{dropped}")
    host_dropped = {
        name: int(value)
        for name, value in metadata.get("camera_host_dropped", {}).items()
    }
    if any(host_dropped.values()):
        errors.append(f"相机主机队列检测到设备序号跳变：{host_dropped}")
    sync_buffer_evicted = {
        name: int(value)
        for name, value in metadata.get("camera_sync_buffer_evicted", {}).items()
    }
    if any(sync_buffer_evicted.values()):
        errors.append(f"同步匹配缓冲区发生淘汰：{sync_buffer_evicted}")
    matcher_summary = dict(metadata.get("sync_matcher", {}))
    unmatched_references = int(matcher_summary.get("unmatched_references", 0))
    if metadata.get("sync_mode") == "fsin" and unmatched_references:
        errors.append(f"FSIN采集存在{unmatched_references}个未匹配cam0触发帧")
    sync_summary.update(
        {
            "matcher": matcher_summary,
            "camera_host_dropped": host_dropped,
            "camera_sync_buffer_evicted": sync_buffer_evicted,
        }
    )
    if metadata.get("status") == "error":
        errors.append(f"采集会话异常结束：{metadata.get('error')}")

    imu_summary = {"enabled": bool(metadata.get("imu", {}).get("enabled"))}
    if imu_summary["enabled"]:
        expected_imu_rate = float(metadata.get("imu", {}).get("rate_hz", 400))
        sensor_rows = {}
        for sensor in ("accel", "gyro"):
            path = session_dir / "imu0" / f"{sensor}.csv"
            try:
                rows = load_imu_rows(path)
            except (KeyError, ValueError, OSError) as exc:
                errors.append(f"{sensor}.csv无效：{exc}")
                continue
            sensor_rows[sensor] = rows
            timestamps = [row.timestamp_ns for row in rows]
            if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
                errors.append(f"{sensor}时间戳未严格递增")
            values = np.asarray([(row.x, row.y, row.z) for row in rows], dtype=float)
            if values.size and not np.all(np.isfinite(values)):
                errors.append(f"{sensor}包含NaN或Inf")
            summary = rate_and_gaps(timestamps, expected_imu_rate)
            summary["axis_std"] = np.std(values, axis=0).tolist() if len(values) else []
            imu_summary[sensor] = summary
            if len(rows) < 2:
                errors.append(f"{sensor}有效样本不足2")
            elif summary["actual_hz"] < expected_imu_rate * 0.8:
                warnings.append(
                    f"{sensor}实际频率{summary['actual_hz']:.1f}低于目标{expected_imu_rate:.1f}"
                )
        if (
            metadata.get("mode") == "imu_calib"
            and len(sensor_rows) == 2
            and all(len(rows) >= 2 for rows in sensor_rows.values())
        ):
            accel_std = np.std([(r.x, r.y, r.z) for r in sensor_rows["accel"]], axis=0)
            gyro_std = np.std([(r.x, r.y, r.z) for r in sensor_rows["gyro"]], axis=0)
            if np.any(accel_std < 1.0) or np.any(gyro_std < 0.1):
                warnings.append("IMU标定序列三轴激励不足")
    elif metadata.get("mode") in ("imu_calib", "vio"):
        errors.append("该模式要求IMU，但session.json标记为未启用")

    if metadata.get("mode") == "response":
        for name, summary in camera_summary.items():
            if len(summary["exposure_values_us"]) < 3:
                warnings.append(f"{name}响应标定曝光档位少于3个")
    if metadata.get("mode") in ("cam_calib", "imu_calib"):
        for name, summary in camera_summary.items():
            if len(summary["exposure_values_us"]) != 1:
                warnings.append(f"{name}标定序列曝光未锁定")
    if metadata.get("mode") in ("response", "cam_calib", "imu_calib"):
        if metadata.get("manual_exposure_us") is None:
            errors.append("标定模式必须关闭自动曝光")
    if metadata.get("codec") == "webp" and metadata.get("mode") != "vio":
        if int(metadata.get("webp_quality") or 0) != 101:
            errors.append("标定序列WebP必须使用101无损质量")

    basalt_ready = (
        not errors
        and not warnings
        and bool(skews)
        and max(skews) <= strict_sync_us
        and metadata.get("status") in ("completed", "interrupted")
    )
    report = {
        "passed": not errors,
        "basalt_ready": basalt_ready,
        "session_dir": str(session_dir.resolve()),
        "mode": metadata.get("mode"),
        "camera_summary": camera_summary,
        "sync": sync_summary,
        "imu": imu_summary,
        "dropped": dropped,
        "errors": errors,
        "warnings": warnings,
    }
    if update_metadata:
        metadata["basalt_ready"] = basalt_ready
        metadata["last_validation"] = {
            "passed": report["passed"],
            "basalt_ready": basalt_ready,
            "errors": len(errors),
            "warnings": len(warnings),
        }
        write_json_atomic(metadata_path, metadata)
    return report


def main(argv=None):
    args = parse_args(argv)
    try:
        report = validate_dataset(
            args.session_dir,
            args.strict_sync_us,
            args.min_groups,
            args.min_aprilgrid_bins,
            not args.skip_image_decode,
        )
    except (RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    report_path = args.report or args.session_dir / "validation_report.json"
    write_json_atomic(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"验证报告：{report_path}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
