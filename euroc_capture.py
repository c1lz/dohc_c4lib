"""Shared color recorder and offline Extended EuRoC finalization.

JSON is used for manifest.yaml: JSON syntax is also valid YAML 1.2.
Hardware imports are lazy so offline export and tests need no OAK device.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CAMERAS = ("cam0", "cam1", "cam2", "cam3")
IMU_HEADER = ["#timestamp [ns]", "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]",
              "w_RS_S_z [rad s^-1]", "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]",
              "a_RS_S_z [m s^-2]"]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader)
        return [row for row in reader if row]


def stamps_checked(rows):
    stamps = [int(row[0]) for row in rows]
    if any(not -(2**63) <= t < 2**63 for t in stamps):
        raise ValueError("时间戳超出int64")
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("时间戳重复或乱序")
    return stamps


def align_imu(accel, gyro, rate):
    """Interpolate accel onto gyro measurement times, never epoch float times."""
    at = stamps_checked(accel)
    gt = stamps_checked(gyro)
    for row in accel + gyro:
        if not all(math.isfinite(float(v)) for v in row[1:4]):
            raise ValueError("IMU包含非有限值")
    output = []
    skipped = 0
    boundary = 0
    for row, t in zip(gyro, gt):
        index = bisect.bisect_left(at, t)
        if index < len(at) and at[index] == t:
            a = [float(v) for v in accel[index][1:4]]
        elif index == 0 or index == len(at):
            boundary += 1
            continue
        elif at[index] - at[index - 1] > 2e9 / rate:
            skipped += 1
            continue
        else:
            weight = (t - at[index - 1]) / (at[index] - at[index - 1])
            a = [(1 - weight) * float(x) + weight * float(y)
                 for x, y in zip(accel[index - 1][1:4], accel[index][1:4])]
        output.append([t, *map(float, row[1:4]), *a])
    return output, {"method": "gyro timestamp; linear accel interpolation; no extrapolation",
                    "boundary_omitted": boundary, "gap_omitted": skipped}


def rate_stats(rows, rate):
    stamps = stamps_checked(rows)
    deltas = [b - a for a, b in zip(stamps, stamps[1:])]
    return {"count": len(rows),
            "mean_rate_hz": (len(rows) - 1) * 1e9 / (stamps[-1] - stamps[0]) if deltas else 0,
            "max_dt_ms": max(deltas) / 1e6 if deltas else None,
            "gaps": sum(d > 1.8e9 / rate for d in deltas)}


def checksums(session):
    target = session / "meta/checksums.sha256"
    with target.open("w", encoding="utf-8") as output:
        for path in sorted(session.rglob("*")):
            if not path.is_file() or path == target:
                continue
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            output.write(f"{digest.hexdigest()}  {path.relative_to(session).as_posix()}\n")


def sync_stats(camera_rows, tolerance_us):
    times = {name: stamps_checked(rows) for name, rows in camera_rows.items()}
    cursors = {name: 0 for name in times}
    skews = []
    for ref in times["cam0"]:
        selected = {}
        for name, stamps in times.items():
            i = bisect.bisect_left(stamps, ref, cursors[name])
            candidates = [j for j in (i - 1, i) if cursors[name] <= j < len(stamps)]
            if not candidates:
                break
            selected[name] = min(candidates, key=lambda j: abs(stamps[j] - ref))
        if len(selected) != len(times):
            continue
        values = [times[n][j] for n, j in selected.items()]
        skew = (max(values) - min(values)) / 1000
        if skew <= tolerance_us:
            skews.append(skew)
            cursors.update({n: j + 1 for n, j in selected.items()})
    ordered = sorted(skews)
    mean = sum(skews) / len(skews) if skews else None
    return {"matched_groups": len(skews), "reference_frames": len(times["cam0"]),
            "unmatched_reference_frames": len(times["cam0"]) - len(skews),
            "mean_us": mean,
            "std_us": math.sqrt(sum((x - mean)**2 for x in skews) / len(skews)) if skews else None,
            "p95_us": ordered[min(len(ordered)-1, math.ceil(.95*len(ordered))-1)] if ordered else None,
            "max_us": max(skews) if skews else None,
            "tolerance_us": tolerance_us}


def finalize(session):
    import cv2
    meta = session / "meta"
    manifest = json.loads((meta / "manifest.yaml").read_text())
    # Recover interrupted finalization after the writer has closed its indexes.
    for name in manifest["cameras"]:
        staged = meta / f"{name}_frames.csv"
        if not staged.exists():
            source = session / "mav0" / name / "data.csv"
            rows = read_rows(source)
            if rows and len(rows[0]) != 5:
                raise ValueError("缺少原始相机索引，不能从EuRoC索引恢复设备序号")
            source.replace(staged)
    for sensor in ("accel", "gyro"):
        staged = meta / "raw_imu" / f"{sensor}.csv"
        if not staged.exists():
            (session / "mav0/imu0" / f"{sensor}.csv").replace(staged)
    errors = list(manifest.get("capture_errors", []))
    if manifest.get("status") not in ("completed", "interrupted"):
        errors.append("采集未正常收尾")
    stats = {"streams": {}, "errors": errors}
    cameras = {}
    for name in manifest["cameras"]:
        rows = read_rows(meta / f"{name}_frames.csv")
        cameras[name] = rows
        try:
            summary = rate_stats(rows, manifest["fps"])
            summary["sequence_gaps"] = sum(max(0, int(b[4])-int(a[4])-1)
                                           for a, b in zip(rows, rows[1:]))
            if summary["sequence_gaps"] or summary["gaps"]:
                errors.append(f"{name}: 设备帧缺口")
            if any(int(b[4]) <= int(a[4]) for a, b in zip(rows, rows[1:])):
                errors.append(f"{name}: 设备序号重复或倒退")
            if not .995 * manifest["fps"] <= summary["mean_rate_hz"] <= 1.005 * manifest["fps"]:
                errors.append(f"{name}: 实际帧率未达目标")
            if len(rows) < manifest["fps"] * max(0, manifest.get("duration_s", 0) - .5):
                errors.append(f"{name}: 帧数不足以覆盖采集时长")
            stats["streams"][name] = summary
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
        expected = {r[1] for r in rows}
        actual = {p.name for p in (session / "mav0" / name / "data").iterdir()}
        if expected != actual or len(expected) != len(rows):
            errors.append(f"{name}: CSV与文件不一致")
        for filename in expected:
            image = cv2.imread(str(session / "mav0" / name / "data" / filename),
                               cv2.IMREAD_UNCHANGED)
            if image is None or image.shape != (800, 1280, 3) or str(image.dtype) != "uint8":
                errors.append(f"{name}: 彩色PNG损坏或尺寸错误 {filename}")
                break
        write_csv(session / "mav0" / name / "data.csv",
                  ["#timestamp [ns]", "filename"], [r[:2] for r in rows])
    try:
        sync = sync_stats(cameras, manifest["max_skew_us"])
        if not sync["matched_groups"] or sync["unmatched_reference_frames"]:
            errors.append("存在未同步配组的参考帧")
    except ValueError as exc:
        sync = {"error": str(exc)}
        errors.append(str(exc))
    write_json(meta / "camera_sync_stats.json", sync)
    accel, gyro = [read_rows(meta / "raw_imu" / f"{s}.csv") for s in ("accel", "gyro")]
    try:
        for name, rows in (("accel", accel), ("gyro", gyro)):
            summary = rate_stats(rows, manifest["imu_rate"])
            stats["streams"][name] = summary
            if summary["gaps"] or not .99*manifest["imu_rate"] <= summary["mean_rate_hz"] <= 1.01*manifest["imu_rate"]:
                errors.append(f"{name}: 采样频率或间隙异常")
            if len(rows) < manifest["imu_rate"] * max(0, manifest.get("duration_s", 0) - .5):
                errors.append(f"{name}: 样本数不足以覆盖采集时长")
        aligned, alignment = align_imu(accel, gyro, manifest["imu_rate"])
        stats["imu_alignment"] = alignment
        if alignment["gap_omitted"] or not aligned:
            errors.append("IMU对齐缺少有效样本或跨越异常间隙")
        write_csv(session / "mav0/imu0/data.csv", IMU_HEADER, aligned)
    except ValueError as exc:
        errors.append(str(exc))
    stats["capture"] = manifest.get("capture_stats", {})
    stats["passed"] = not errors
    stats["calibration_ready"] = False
    stats["warnings"] = [
        "曝光时间戳参考时刻尚未确认；必须确认后再解释Camera–IMU时间偏移",
        "QA检查数据完整性，画质/动作激励和Mocap时空标定需另行验收"]
    stats["hardware_performance_verified"] = False
    manifest["qa"] = "REJECTED" if errors else "QA_PASSED"
    manifest["qualification_note"] = "Dataset QA only; not a long-duration hardware qualification"
    write_json(meta / "recorder_stats.json", stats)
    write_json(meta / "manifest.yaml", manifest)
    checksums(session)
    return stats


def import_mocap(source, session, clock_domain):
    """Import project-standard CSV; retain source bytes, timestamps and invalid poses."""
    rows = read_rows(source)
    stamps_checked(rows)
    if not rows:
        raise ValueError("Mocap文件为空")
    with source.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    standard = ["#timestamp [ns]", "p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]",
                "q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []"]
    compact = ["timestamp_ns", "px", "py", "pz", "qw", "qx", "qy", "qz"]
    if header[:8] not in (standard, compact):
        raise ValueError("Mocap表头必须明确为timestamp_ns,px,py,pz,qw,qx,qy,qz或项目EuRoC标准表头")
    if header[8:] not in ([], ["tracking_valid"]):
        raise ValueError("第九列仅支持tracking_valid")
    invalid = 0
    for row in rows:
        if len(row) != len(header) or not all(math.isfinite(float(v)) for v in row[1:8]):
            raise ValueError("Mocap行长度或数值无效")
        valid = int(row[8]) if len(row) == 9 else 1
        if valid not in (0, 1):
            raise ValueError("tracking_valid必须为0或1")
        invalid += 1 - valid
        norm = math.sqrt(sum(float(v)**2 for v in row[4:8]))
        if valid and abs(norm - 1) > .01:
            raise ValueError("有效Mocap四元数范数偏离1超过0.01")
    meta = session / "meta"
    manifest = json.loads((meta / "manifest.yaml").read_text())
    destination = session / "mav0/mocap0"
    destination.mkdir(exist_ok=False)
    shutil.copyfile(source, meta / "mocap_source.csv")
    write_csv(destination / "data.csv", standard + ["tracking_valid"],
              [r if len(r) == 9 else r + ["1"] for r in rows])
    manifest["mocap"] = {"clock_domain": clock_domain, "timestamp_reference": "mocap_measurement_time",
                         "pose_definition": "T_mocap_world_mocap0", "quaternion_order": "wxyz",
                         "position_unit": "m", "time_alignment": "NOT_CALIBRATED",
                         "samples": len(rows), "tracking_invalid": invalid,
                         "tracking_valid_ratio": 1-invalid/len(rows)}
    write_json(meta / "manifest.yaml", manifest)
    checksums(session)


def parse_args(camera_ids, argv=None):
    p = argparse.ArgumentParser(description="彩色PNG + RAW IMU Extended EuRoC采集")
    p.add_argument("--output-dir", type=Path, default=Path("euroc_datasets"))
    p.add_argument("--duration", type=float, default=90 if len(camera_ids) == 1 else None)
    p.add_argument("--fps", type=float, default=30)
    p.add_argument("--imu-rate", type=int, default=400)
    p.add_argument("--sync-mode", choices=("fsin", "free-run"), default="fsin")
    p.add_argument("--max-skew-us", type=float, default=1000)
    p.add_argument("--manual-exposure-us", type=int, default=2000)
    p.add_argument("--iso", type=int, default=400)
    p.add_argument("--manual-focus", type=int)
    p.add_argument("--queue-size", type=int, default=256)
    p.add_argument("--image-workers-per-camera", type=int, default=2)
    p.add_argument("--warmup-seconds", type=float, default=2)
    p.add_argument("--allow-usb2", action="store_true", help="诊断用途；默认强制USB3")
    p.add_argument("--min-tags", type=int, default=6)
    p.add_argument("--min-sharpness", type=float, default=80)
    p.add_argument("--min-brightness", type=float, default=45)
    p.add_argument("--max-brightness", type=float, default=220)
    p.add_argument("--max-clipped-percent", type=float, default=30)
    p.add_argument("--min-area-ratio", type=float, default=.025)
    p.add_argument("--max-area-ratio", type=float, default=.65)
    p.add_argument("--quality-hz", type=float, default=2)
    p.add_argument("--session-dir", type=Path)
    p.add_argument("--finalize-only", action="store_true")
    if len(camera_ids) == 1:
        p.add_argument("--import-mocap", type=Path)
        p.add_argument("--mocap-clock-domain", default="mocap_server_clock")
    a = p.parse_args(argv)
    if any(not math.isfinite(v) or v <= 0 for v in
           (a.fps, a.imu_rate, a.max_skew_us, a.manual_exposure_us, a.iso, a.quality_hz)):
        p.error("频率、曝光、ISO和同步阈值必须为有限正数")
    if a.duration is not None and (not math.isfinite(a.duration) or a.duration <= 0):
        p.error("duration必须为有限正数")
    if a.queue_size < 4 or a.image_workers_per_camera < 1 or not math.isfinite(a.warmup_seconds) or a.warmup_seconds < 0:
        p.error("无效队列、线程数或预热时间")
    if a.manual_focus is not None and not 0 <= a.manual_focus <= 255:
        p.error("focus必须为0..255")
    if not (1 <= a.min_tags <= 36 and 0 <= a.min_brightness <= a.max_brightness <= 255
            and 0 <= a.max_clipped_percent <= 100 and math.isfinite(a.min_sharpness)
            and a.min_sharpness >= 0 and 0 < a.min_area_ratio < a.max_area_ratio <= 1):
        p.error("无效画质门槛")
    importing = getattr(a, "import_mocap", None)
    if (importing or a.finalize_only) and a.session_dir is None:
        p.error("离线操作需要--session-dir")
    if importing and a.finalize_only:
        p.error("导入和收尾请分别执行")
    return a


def capture(a, camera_ids):
    import cv2
    import depthai as dai
    from dohc2_c4lib.aprilgrid_common import CAMERA_MAPPING, SENSOR_RESOLUTION, create_detector, quality_reasons
    from dohc2_c4lib.guided_capture import assess_guided, thresholds_from_args
    from dohc2_c4lib.record_oak4p import DatasetWriter, ImageTask, ImuTask, CameraFrameState
    from dohc2_c4lib.oak4p_dataset import timedelta_ns, CAMERA_CSV_FIELDS, IMU_CSV_FIELDS

    session = None
    with dai.Device() as device, dai.Pipeline(device) as pipeline:
        available = {s.name: s for s in device.getConnectedCameras()}
        missing = [CAMERA_MAPPING[n] for n in camera_ids if CAMERA_MAPPING[n] not in available]
        if missing:
            raise RuntimeError(f"缺少相机接口: {missing}")
        queues = {}
        for name in camera_ids:
            camera = pipeline.create(dai.node.Camera)
            camera.setSensorType(dai.CameraSensorType.COLOR)
            camera.build(available[CAMERA_MAPPING[name]], sensorResolution=SENSOR_RESOLUTION, sensorFps=a.fps)
            if a.sync_mode == "fsin":
                camera.initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)
            camera.initialControl.setManualExposure(a.manual_exposure_us, a.iso)
            if a.manual_focus is not None:
                camera.initialControl.setManualFocus(a.manual_focus)
            output = camera.requestFullResolutionOutput(type=dai.ImgFrame.Type.NV12, fps=a.fps)
            queues[name] = output.createOutputQueue(maxSize=max(8, a.queue_size//len(camera_ids)), blocking=False)
        imu = pipeline.create(dai.node.IMU)
        imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], a.imu_rate)
        imu.setBatchReportThreshold(10)
        imu.setMaxBatchReports(20)
        imu_queue = imu.out.createOutputQueue(maxSize=a.queue_size, blocking=False)
        pipeline.start()
        if not a.allow_usb2 and "SUPER" not in str(device.getUsbSpeed()).upper():
            raise RuntimeError("USB3 SuperSpeed未协商成功")
        deadline = time.monotonic() + a.warmup_seconds
        while time.monotonic() < deadline:
            for q in [*queues.values(), imu_queue]:
                q.tryGetAll()
            time.sleep(.005)
        started = datetime.now().astimezone()
        label = "cam0_imu" if len(camera_ids) == 1 else "fourcam_imu"
        session = a.output_dir / started.strftime(f"{label}_%Y%m%d_%H%M%S_%f")
        meta = session / "meta"
        (meta / "raw_imu").mkdir(parents=True)
        (session / "mav0/imu0").mkdir(parents=True)
        # Writer stages extended indexes separately; finalization emits EuRoC indexes.
        for name in camera_ids:
            (session / "mav0" / name / "data").mkdir(parents=True)
            write_csv(session / "mav0" / name / "data.csv", CAMERA_CSV_FIELDS, [])
        for sensor in ("accel", "gyro"):
            write_csv(session / "mav0/imu0" / f"{sensor}.csv", IMU_CSV_FIELDS, [])
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
        target = Path(__file__).resolve().parent / "configs/target.yaml"
        if target.is_file():
            shutil.copyfile(target, meta / "aprilgrid.yaml")
        manifest = {"format": {"name": "extended_euroc", "version": 1},
                    "cameras": list(camera_ids), "camera_mapping": {n: CAMERA_MAPPING[n] for n in camera_ids},
                    "fps": a.fps, "imu_rate": a.imu_rate, "max_skew_us": a.max_skew_us,
                    "config": config, "resolution": [1280, 800], "image_format": "color PNG",
                    "image_pipeline": "COLOR ISP -> NV12 -> getCvFrame BGR -> PNG compression 3",
                    "depthai_version": dai.__version__, "opencv_version": cv2.__version__,
                    "recorder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "device_id": device.getDeviceId(), "imu_model": str(device.getConnectedIMU()),
                    "imu_units": {"gyro": "rad/s", "accel": "m/s2"},
                    "imu_axes": "device native axes; physical rig directions not surveyed",
                    "target_config": "meta/aprilgrid.yaml; verify physical print dimensions",
                    "focus": a.manual_focus if a.manual_focus is not None else "device default; operator must confirm fixed lens",
                    "white_balance": "device default; not explicitly locked",
                    "usb_speed": str(device.getUsbSpeed()), "started_at": started.isoformat(),
                    "timestamp": {"unit": "int64 ns", "clock_domain": "oak_device_clock",
                                  "camera_reference": "UNKNOWN: getTimestampDevice exposure reference unverified",
                                  "imu_reference": "measurement_time", "host_time_used_for_measurements": False},
                    "status": "running", "qa": "RAW"}
        write_json(meta / "manifest.yaml", manifest)
        writer = DatasetWriter(session / "mav0", "png", 101, a.queue_size,
                               a.image_workers_per_camera, camera_ids=camera_ids, color=True)
        states = {n: CameraFrameState(1) for n in camera_ids}
        stop = threading.Event()
        quality_stop = threading.Event()
        errors = []
        latest = {}
        latest_lock = threading.Lock()
        last_received = {n: time.monotonic() for n in (*camera_ids, "imu")}
        imu_counts = {"accel": 0, "gyro": 0}
        last_camera_stamp = {}
        last_camera_sequence = {}
        events = []
        start_clock = time.monotonic()
        received_counts = {n: 0 for n in camera_ids}
        threads = []

        def receive_camera(name):
            try:
                while not stop.is_set():
                    for frame in queues[name].tryGetAll():
                        last_received[name] = time.monotonic()
                        received_counts[name] += 1
                        sample = states[name].consume(frame, retain_for_matching=False)
                        if (sample.timestamp_ns <= last_camera_stamp.get(name, -1)
                                or sample.sequence_num <= last_camera_sequence.get(name, -1)):
                            raise RuntimeError("设备时间戳或序号重复/倒退，已停止避免覆盖图像")
                        last_camera_stamp[name] = sample.timestamp_ns
                        last_camera_sequence[name] = sample.sequence_num
                        if not writer.submit(ImageTask(name, sample.timestamp_ns, sample.sequence_num,
                                                       sample.exposure_us, sample.iso, frame)):
                            raise RuntimeError(f"{name}: 写盘队列溢出")
                        with latest_lock:
                            latest[name] = (sample.timestamp_ns, frame)
                    stop.wait(.001)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                stop.set()

        def receive_imu():
            try:
                while not stop.is_set():
                    for message in imu_queue.tryGetAll():
                        last_received["imu"] = time.monotonic()
                        for packet in message.packets:
                            for sensor, report in (("accel", packet.acceleroMeter), ("gyro", packet.gyroscope)):
                                imu_counts[sensor] += 1
                                if not writer.submit(ImuTask(sensor, timedelta_ns(report.getTimestampDevice()),
                                                            int(message.getSequenceNum()), report.x, report.y, report.z)):
                                    raise RuntimeError("IMU写盘队列溢出")
                    stop.wait(.001)
            except Exception as exc:
                errors.append(f"imu: {exc}")
                stop.set()

        def quality_worker():
            try:
                detector = create_detector()
                seen = {}
                with (meta / "image_quality.csv").open("w", newline="") as handle:
                    csvout = csv.writer(handle)
                    csvout.writerow(["timestamp_ns", "camera", "tags", "sharpness", "brightness",
                                     "clipped_percent", "area_ratio", "reasons"])
                    while not quality_stop.is_set():
                        with latest_lock:
                            snapshots = dict(latest)
                        for name, (stamp, frame) in snapshots.items():
                            if seen.get(name) == stamp:
                                continue
                            seen[name] = stamp
                            assessment = assess_guided(frame.getCvFrame(), detector, None, time.monotonic(), 4)
                            base = assessment.base
                            reasons = quality_reasons(base, thresholds_from_args(a))
                            if not a.min_area_ratio <= assessment.area_ratio <= a.max_area_ratio:
                                reasons.append("board area")
                            csvout.writerow([stamp, name, len(base.tag_ids), base.sharpness,
                                             base.brightness, base.clipped_percent, assessment.area_ratio,
                                             "|".join(reasons)])
                            if len(camera_ids) == 1:
                                print(f"画质 {name}: {'GOOD' if not reasons else ', '.join(reasons)}", flush=True)
                        handle.flush()
                        quality_stop.wait(1/a.quality_hz)
            except Exception as exc:
                errors.append(f"quality: {exc}")
                stop.set()

        try:
            writer.start()
            threads = [threading.Thread(target=receive_camera, args=(n,)) for n in camera_ids]
            threads += [threading.Thread(target=receive_imu), threading.Thread(target=quality_worker)]
            for thread in threads:
                thread.start()
            events.append([0, "START", "host monotonic relative ns"])
            print(f"数据目录: {session.resolve()}", flush=True)
            previous_counts = {n: 0 for n in (*camera_ids, "accel", "gyro")}
            last_dashboard = start_clock
            while not stop.wait(1):
                elapsed = time.monotonic() - start_clock
                if writer.error:
                    raise RuntimeError(f"写盘失败: {writer.error}")
                stalled = [n for n, last in last_received.items() if time.monotonic()-last > 5]
                if stalled:
                    raise RuntimeError(f"超过5秒无数据: {stalled}")
                if not pipeline.isRunning():
                    raise RuntimeError("设备流水线意外停止")
                current_counts = {**received_counts, **imu_counts}
                dashboard_time = time.monotonic()
                rates = {n: round((count-previous_counts[n])/(dashboard_time-last_dashboard), 1)
                         for n, count in current_counts.items()}
                previous_counts, last_dashboard = current_counts, dashboard_time
                print(f"{elapsed:.0f}s rates={rates} written={writer.written} queue={writer.queue_depth} "
                      f"host_gaps={sum(s.host_dropped for s in states.values())}", flush=True)
                if a.duration is not None and elapsed >= a.duration:
                    break
            manifest["status"] = "completed" if not errors else "error"
        except KeyboardInterrupt:
            manifest["status"] = "interrupted"
        except Exception as exc:
            manifest["status"] = "error"
            errors.append(str(exc))
        finally:
            capture_duration = time.monotonic()-start_clock
            stop.set()
            quality_stop.set()
            for thread in threads:
                if thread.ident is not None:
                    thread.join()
            try:
                pipeline.stop()
            except Exception as exc:
                errors.append(f"pipeline stop: {exc}")
            try:
                writer.close()
            except Exception as exc:
                errors.append(str(exc))
            duration = capture_duration
            events.append([int(duration*1e9), "STOP", "host monotonic relative ns"])
            write_csv(meta / "events.csv", ["timestamp_ns", "event", "clock"], events)
            manifest.update(ended_at=datetime.now().astimezone().isoformat(), duration_s=duration,
                            capture_errors=errors, capture_stats={
                                "written": writer.written, "writer_dropped": writer.dropped,
                                "queue_high_watermark": writer.high_watermark,
                                "host_sequence_gaps": {n: s.host_dropped for n, s in states.items()}})
            if errors:
                manifest["status"] = "error"
            write_json(meta / "manifest.yaml", manifest)
            for name in camera_ids:
                (session / "mav0" / name / "data.csv").replace(meta / f"{name}_frames.csv")
            for sensor in ("accel", "gyro"):
                (session / "mav0/imu0" / f"{sensor}.csv").replace(meta / "raw_imu" / f"{sensor}.csv")
        report = finalize(session)
        print(f"数据: {session.resolve()}\nQA: {'PASS' if report['passed'] else 'REJECTED'}", flush=True)
        return 0 if report["passed"] else 2


def main(camera_ids, argv=None):
    a = parse_args(camera_ids, argv)
    try:
        if getattr(a, "import_mocap", None):
            import_mocap(a.import_mocap, a.session_dir, a.mocap_clock_domain)
            print("Mocap已导入；尚未做时钟对齐或外参求解。")
            return 0
        if a.finalize_only:
            report = finalize(a.session_dir)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["passed"] else 2
        return capture(a, camera_ids)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
