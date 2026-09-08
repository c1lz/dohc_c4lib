#!/usr/bin/env python3
"""Continuously record four OAK-4P cameras and its onboard IMU."""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

if Path("/usr/share/fonts/truetype/dejavu").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import depthai as dai
import numpy as np

# Support both ``python -m dohc2_c4lib.record_oak4p`` and direct execution of
# this file from its package directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dohc2_c4lib.aprilgrid_common import (
    CAMERA_IDS,
    CAMERA_MAPPING,
    SENSOR_RESOLUTION,
    create_detector,
    detect_aprilgrid,
    write_json_atomic,
)
from dohc2_c4lib.frame_sync import TimestampGroupMatcher, group_skew_us
from dohc2_c4lib.oak4p_dataset import (
    CAMERA_CSV_FIELDS,
    DEFAULT_MAX_PIXEL_PHASE_SPREAD,
    IMU_CSV_FIELDS,
    build_groups,
    load_camera_rows,
    pixel_phase_metrics,
    timedelta_ns,
    write_groups,
)


WINDOW_NAME = "OAK-4P Continuous Recorder"
MODE_DEFAULTS = {
    "response": {"codec": "png", "manual_exposure_us": 2_000, "skip_frames": 4},
    "cam_calib": {"codec": "png", "manual_exposure_us": 2_000, "skip_frames": 4},
    "imu_calib": {"codec": "webp", "manual_exposure_us": 2_000, "skip_frames": 0},
    "vio": {"codec": "webp", "manual_exposure_us": 2_000, "skip_frames": 0},
}


@dataclass
class ImageTask:
    camera_id: str
    timestamp_ns: int
    sequence_num: int
    exposure_us: int
    iso: int
    image: object


@dataclass
class ImuTask:
    sensor: str
    timestamp_ns: int
    sequence_num: int
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class CameraFrameSample:
    sequence_num: int
    timestamp_ns: int
    exposure_us: int
    iso: int
    frame: object


class CameraFrameState:
    """Keep a bounded set of full-resolution frames awaiting sync matching."""

    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.pending: OrderedDict[int, CameraFrameSample] = OrderedDict()
        self.last_sequence: int | None = None
        self.host_dropped = 0
        self.buffer_evicted = 0

    def consume(self, frame, *, retain_for_matching: bool = True) -> CameraFrameSample:
        sequence_num = int(frame.getSequenceNum())
        if self.last_sequence is not None and sequence_num > self.last_sequence + 1:
            self.host_dropped += sequence_num - self.last_sequence - 1
        self.last_sequence = sequence_num
        sample = CameraFrameSample(
            sequence_num=sequence_num,
            timestamp_ns=timedelta_ns(frame.getTimestampDevice()),
            exposure_us=int(
                frame.getExposureTime().total_seconds() * 1_000_000
            ),
            iso=int(frame.getSensitivity()),
            frame=frame,
        )
        if retain_for_matching:
            self.pending[sequence_num] = sample
            while len(self.pending) > self.buffer_size:
                self.pending.popitem(last=False)
                self.buffer_evicted += 1
        return sample


def parse_exposure_sweep(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("曝光序列必须为逗号分隔的整数微秒") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("曝光序列必须包含正整数")
    return values


def should_record_group(matched_group_number: int, skip_frames: int) -> bool:
    """Apply frame skipping once per synchronized group, never per camera."""
    if matched_group_number < 1 or skip_frames < 0:
        raise ValueError("matched_group_number must be >=1 and skip_frames >=0")
    return (matched_group_number - 1) % (skip_frames + 1) == 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="连续采集OAK-4P四相机和板载IMU")
    parser.add_argument("--mode", choices=tuple(MODE_DEFAULTS), required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/mnt/sdcard/oak4p_datasets")
    )
    parser.add_argument("--sync-mode", choices=("free-run", "fsin"), default="free-run")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--imu-rate", type=int, default=400)
    parser.add_argument("--max-skew-us", type=int)
    parser.add_argument("--strict-sync-us", type=int, default=1_000)
    parser.add_argument(
        "--codec",
        choices=(
            "png", "webp", "gray8", "nv12", "mjpeg-lossless", "mjpeg-q100", "mjpeg-q95"
        ),
        help="落盘格式；nv12保留设备输出的满分辨率原始彩色NV12字节，不解码、不压缩",
    )
    parser.add_argument(
        "--camera-transfer-format",
        choices=("nv12", "gray8", "mjpeg-lossless", "mjpeg-q100", "mjpeg-q95"),
        default="nv12",
        help="设备到主机的帧格式；gray8可比NV12减少约1/3 USB带宽",
    )
    parser.add_argument(
        "--require-usb3",
        action="store_true",
        help="流水线启动后若OAK未协商到SuperSpeed则在建数据目录前失败",
    )
    parser.add_argument(
        "--start-on-aprilgrid",
        action="store_true",
        help="任一路首次检测到目标AprilGrid后才创建会话并开始计时",
    )
    parser.add_argument(
        "--start-min-tags",
        type=int,
        default=1,
        help="启动采集所需的最少有效AprilGrid标签数",
    )
    parser.add_argument(
        "--start-timeout",
        type=float,
        default=0.0,
        help="等待AprilGrid的超时秒数；0表示无限等待",
    )
    parser.add_argument("--webp-quality", type=int, default=101)
    parser.add_argument("--manual-exposure-us", type=int)
    parser.add_argument("--iso", type=int, default=400)
    parser.add_argument("--manual-focus", type=int)
    parser.add_argument("--skip-frames", type=int)
    parser.add_argument(
        "--capture-strategy",
        choices=("sync-group", "stream-first"),
        default="sync-group",
        help=(
            "sync-group在主线程实时配组后写盘；stream-first用独立线程持续排空每路"
            "设备队列，结束后按设备时间戳配组，适合算力受限主机"
        ),
    )
    parser.add_argument("--queue-size", type=int, default=128)
    parser.add_argument(
        "--image-workers-per-camera",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4) // len(CAMERA_IDS))),
        help="每路相机的并行图像编码/写入线程数",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, help="可选自动停止秒数")
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=1.0,
        help="正式记录前持续排空设备队列的预热时间",
    )
    parser.add_argument(
        "--max-pixel-phase-spread",
        type=float,
        default=DEFAULT_MAX_PIXEL_PHASE_SPREAD,
        help="采集前允许的最大2x2像素相位均值差；用于拒绝未去马赛克图像",
    )
    parser.add_argument(
        "--exposure-sweep-us",
        type=parse_exposure_sweep,
        default=(250, 500, 1_000, 2_000, 4_000, 8_000),
    )
    parser.add_argument("--frames-per-exposure", type=int, default=20)
    args = parser.parse_args(argv)
    defaults = MODE_DEFAULTS[args.mode]
    args.codec = args.codec or defaults["codec"]
    if args.manual_exposure_us is None:
        args.manual_exposure_us = defaults["manual_exposure_us"]
    if args.skip_frames is None:
        args.skip_frames = defaults["skip_frames"]
    if args.max_skew_us is None:
        args.max_skew_us = 1_000 if args.sync_mode == "fsin" else 20_000
    if args.mode == "response":
        args.manual_exposure_us = args.exposure_sweep_us[0]
    if args.capture_strategy == "stream-first" and args.skip_frames != 0:
        parser.error("stream-first仅支持skip-frames=0，避免各路独立抽帧破坏同步")
    if args.capture_strategy == "stream-first" and not args.headless:
        parser.error("stream-first要求--headless，避免显示路径干扰持续接收")
    mjpeg_codecs = {"mjpeg-lossless", "mjpeg-q100", "mjpeg-q95"}
    if (
        (args.codec in mjpeg_codecs) !=
        (args.camera_transfer_format in mjpeg_codecs)
        or (
            args.codec in mjpeg_codecs
            and args.codec != args.camera_transfer_format
        )
    ):
        parser.error("MJPEG存储与传输格式必须以相同模式同时启用")
    if args.fps <= 0 or args.imu_rate <= 0 or args.max_skew_us <= 0:
        parser.error("fps、imu-rate和max-skew-us必须大于0")
    if (
        args.queue_size < 4
        or args.skip_frames < 0
        or args.frames_per_exposure < 1
        or args.image_workers_per_camera < 1
        or args.warmup_seconds < 0
        or args.max_pixel_phase_spread <= 0
        or args.start_min_tags < 1
        or args.start_timeout < 0
    ):
        parser.error(
            "queue-size必须>=4，skip-frames>=0，frames-per-exposure>=1，"
            "image-workers-per-camera>=1，warmup-seconds>=0，"
            "max-pixel-phase-spread>0，start-min-tags>=1，start-timeout>=0"
        )
    if not 1 <= args.webp_quality <= 101:
        parser.error("webp-quality必须为1..101，101表示无损")
    if args.manual_focus is not None and not 0 <= args.manual_focus <= 255:
        parser.error("manual-focus必须为0..255")
    return args


def validate_camera_mapping(sockets):
    by_name = {socket.name: socket for socket in sockets}
    missing = [socket for socket in CAMERA_MAPPING.values() if socket not in by_name]
    if missing:
        raise RuntimeError(f"缺少相机接口：{missing}")
    return {name: by_name[socket] for name, socket in CAMERA_MAPPING.items()}


class DatasetWriter:
    def __init__(
        self,
        session_dir: Path,
        codec: str,
        webp_quality: int,
        maxsize: int,
        image_workers_per_camera: int = 1,
        camera_ids=CAMERA_IDS,
        color: bool = False,
    ):
        self.camera_ids = tuple(camera_ids)
        self.color = color
        self.session_dir = session_dir
        self.codec = codec
        self.webp_quality = webp_quality
        self.image_workers_per_camera = image_workers_per_camera
        image_queue_size = max(4, maxsize // len(self.camera_ids))
        self.queues = {
            name: queue.Queue(maxsize=image_queue_size) for name in self.camera_ids
        }
        self.queues["imu"] = queue.Queue(maxsize=maxsize)
        self.dropped = {name: 0 for name in (*self.camera_ids, "accel", "gyro")}
        self.written = {name: 0 for name in (*self.camera_ids, "accel", "gyro")}
        self.error: BaseException | None = None
        self.high_watermark = 0
        self.high_watermark_by_queue = {name: 0 for name in self.queues}
        self._stop = object()
        self._error_lock = threading.Lock()
        self._started = False
        self._submit_order = {name: 0 for name in self.camera_ids}
        self._camera_states = {
            name: {
                "condition": threading.Condition(),
                "completed": {},
                "next_commit": 0,
                "handle": None,
                "writer": None,
            }
            for name in self.camera_ids
        }
        self._imu_handles = {}
        self._imu_writers = {}
        self.threads = {
            name: [
                threading.Thread(
                    target=self._run_camera,
                    args=(name,),
                    name=f"oak4p-writer-{name}-{worker_index}",
                    daemon=True,
                )
                for worker_index in range(image_workers_per_camera)
            ]
            for name in self.camera_ids
        }
        self.threads["imu"] = [
            threading.Thread(
                target=self._run_imu,
                name="oak4p-writer-imu",
                daemon=True,
            )
        ]

    def start(self):
        try:
            for name in self.camera_ids:
                handle = (self.session_dir / name / "data.csv").open(
                    "a", newline="", encoding="utf-8"
                )
                self._camera_states[name]["handle"] = handle
                self._camera_states[name]["writer"] = csv.writer(handle)
            for sensor in ("accel", "gyro"):
                handle = (self.session_dir / "imu0" / f"{sensor}.csv").open(
                    "a", newline="", encoding="utf-8"
                )
                self._imu_handles[sensor] = handle
                self._imu_writers[sensor] = csv.writer(handle)
            for stream_threads in self.threads.values():
                for thread in stream_threads:
                    thread.start()
            self._started = True
        except BaseException:
            self._close_handles()
            raise

    @property
    def queue_depth(self) -> int:
        return sum(output_queue.qsize() for output_queue in self.queues.values())

    @property
    def queue_capacity(self) -> int:
        return sum(output_queue.maxsize for output_queue in self.queues.values())

    @property
    def queue_depths(self) -> dict[str, int]:
        return {name: output_queue.qsize() for name, output_queue in self.queues.items()}

    def submit(self, task) -> bool:
        stream = task.camera_id if isinstance(task, ImageTask) else task.sensor
        queue_name = task.camera_id if isinstance(task, ImageTask) else "imu"
        output_queue = self.queues[queue_name]
        queued_task = task
        if isinstance(task, ImageTask):
            queued_task = (self._submit_order[queue_name], task)
        try:
            output_queue.put_nowait(queued_task)
            if isinstance(task, ImageTask):
                self._submit_order[queue_name] += 1
            depth = output_queue.qsize()
            self.high_watermark_by_queue[queue_name] = max(
                self.high_watermark_by_queue[queue_name], depth
            )
            self.high_watermark = max(self.high_watermark, self.queue_depth)
            return True
        except queue.Full:
            self.dropped[stream] += 1
            return False

    def submit_camera_group(self, tasks: dict[str, ImageTask]) -> bool:
        """Atomically enqueue one synchronized four-camera group."""
        if set(tasks) != set(self.camera_ids):
            raise ValueError("camera group must contain each configured camera once")
        if any(self.queues[name].full() for name in self.camera_ids):
            for name in self.camera_ids:
                self.dropped[name] += 1
            return False
        for name in self.camera_ids:
            if not self.submit(tasks[name]):
                raise RuntimeError("camera group enqueue lost atomicity")
        return True

    def close(self):
        if not self._started:
            self._close_handles()
            return
        for name, output_queue in self.queues.items():
            for thread in self.threads[name]:
                while thread.is_alive():
                    try:
                        output_queue.put(self._stop, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        for stream_threads in self.threads.values():
            for thread in stream_threads:
                thread.join()
        self._close_handles()
        self._started = False
        if self.error:
            raise RuntimeError(f"写盘线程失败：{self.error}") from self.error

    def _close_handles(self):
        handles = [
            state["handle"]
            for state in self._camera_states.values()
            if state["handle"] is not None
        ]
        handles.extend(self._imu_handles.values())
        for handle in handles:
            handle.flush()
            handle.close()
        for state in self._camera_states.values():
            state["handle"] = None
            state["writer"] = None
        self._imu_handles.clear()
        self._imu_writers.clear()

    def _set_error(self, exc: BaseException):
        with self._error_lock:
            if self.error is None:
                self.error = exc

    def _run_camera(self, camera_id: str):
        try:
            output_queue = self.queues[camera_id]
            state = self._camera_states[camera_id]
            params = {
                "png": [cv2.IMWRITE_PNG_COMPRESSION, 3],
                "webp": [cv2.IMWRITE_WEBP_QUALITY, self.webp_quality],
            }.get(self.codec)
            while True:
                queued_task = output_queue.get()
                if queued_task is self._stop:
                    break
                order, task = queued_task
                suffix = "jpg" if self.codec.startswith("mjpeg-") else self.codec
                filename = f"{task.timestamp_ns}.{suffix}"
                path = self.session_dir / camera_id / "data" / filename
                if self.codec.startswith("mjpeg-"):
                    encoded = np.asarray(task.image.getData(), dtype=np.uint8)
                    encoded.tofile(path)
                elif self.codec == "nv12":
                    if isinstance(task.image, np.ndarray):
                        raise RuntimeError("NV12落盘要求DepthAI原始帧")
                    encoded = np.asarray(task.image.getData(), dtype=np.uint8)
                    expected_size = SENSOR_RESOLUTION[0] * SENSOR_RESOLUTION[1] * 3 // 2
                    if encoded.size != expected_size:
                        raise RuntimeError(
                            f"NV12帧长度为{encoded.size}，应为{expected_size}"
                        )
                    encoded.tofile(path)
                else:
                    image = (
                        task.image
                        if isinstance(task.image, np.ndarray)
                        else (task.image.getCvFrame() if self.color else frame_to_gray(task.image))
                    )
                    if self.color and (
                        image.dtype != np.uint8 or image.shape != (800, 1280, 3)
                    ):
                        raise RuntimeError("彩色输出必须为1280x800 uint8 BGR")
                if self.codec == "gray8":
                    if image.dtype != np.uint8 or image.ndim != 2:
                        raise RuntimeError(
                            f"GRAY8要求uint8单通道图像：{image.dtype}/{image.shape}"
                        )
                    image.tofile(path)
                elif self.codec != "nv12" and not self.codec.startswith("mjpeg-") and not cv2.imwrite(
                    str(path), image, params
                ):
                    raise RuntimeError(f"无法写入图像：{path}")
                row = (
                    task.timestamp_ns,
                    filename,
                    task.exposure_us,
                    task.iso,
                    task.sequence_num,
                )
                with state["condition"]:
                    state["completed"][order] = row
                    while state["next_commit"] in state["completed"]:
                        next_row = state["completed"].pop(state["next_commit"])
                        state["writer"].writerow(next_row)
                        state["next_commit"] += 1
                        self.written[camera_id] += 1
        except BaseException as exc:
            self._set_error(exc)

    def _run_imu(self):
        try:
            output_queue = self.queues["imu"]
            while True:
                task = output_queue.get()
                if task is self._stop:
                    break
                self._imu_writers[task.sensor].writerow(
                    (task.timestamp_ns, task.x, task.y, task.z, task.sequence_num)
                )
                self.written[task.sensor] += 1
        except BaseException as exc:
            self._set_error(exc)


def create_session(
    args,
    device_id: str,
    depthai_version: str,
    camera_sensor_names=None,
    imu_model=None,
) -> tuple[Path, dict]:
    started = datetime.now().astimezone()
    session_dir = args.output_dir / started.strftime(f"{args.mode}_%Y%m%d_%H%M%S_%f")
    for name in CAMERA_IDS:
        (session_dir / name / "data").mkdir(parents=True)
        with (session_dir / name / "data.csv").open("w", newline="", encoding="utf-8") as output:
            csv.writer(output).writerow(CAMERA_CSV_FIELDS)
    (session_dir / "imu0").mkdir()
    for sensor in ("accel", "gyro"):
        with (session_dir / "imu0" / f"{sensor}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as output:
            csv.writer(output).writerow(IMU_CSV_FIELDS)
    write_groups(session_dir / "groups.csv", [])
    metadata = {
        "schema_version": 1,
        "started_at": started.isoformat(timespec="microseconds"),
        "status": "running",
        "mode": args.mode,
        "device_id": device_id,
        "depthai_version": depthai_version,
        "camera_mapping": CAMERA_MAPPING,
        "camera_sensor_names": camera_sensor_names or {},
        "sensor_resolution": list(SENSOR_RESOLUTION),
        "sensor_fps": args.fps,
        "camera_sensor_type": "COLOR",
        "camera_output_type": args.camera_transfer_format.upper(),
        "stored_image_type": (
            (
                "LOSSLESS_JPEG"
                if args.codec == "mjpeg-lossless"
                else args.codec.replace("mjpeg-", "JPEG_").upper()
            )
            if args.codec.startswith("mjpeg-")
            else ("NV12" if args.codec == "nv12" else "GRAY8")
        ),
        "gray_conversion": (
            (
                "POSTHOC_LOSSLESS_JPEG_DECODE_TO_GRAY"
                if args.camera_transfer_format == "mjpeg-lossless"
                else (
                    "POSTHOC_JPEG_Q95_DECODE_TO_GRAY"
                    if args.camera_transfer_format == "mjpeg-q95"
                    else "POSTHOC_JPEG_Q100_DECODE_TO_GRAY"
                )
            )
            if args.camera_transfer_format.startswith("mjpeg-")
            else (
                "DEVICE_ISP_GRAY8"
                if args.camera_transfer_format == "gray8"
                else (
                    "NONE_RAW_NV12_STORED"
                    if args.codec == "nv12"
                    else "NV12_Y_PLANE"
                )
            )
        ),
        "device_jpeg_lossless": args.codec == "mjpeg-lossless",
        "device_jpeg_quality": (
            95 if args.codec == "mjpeg-q95" else (100 if args.codec == "mjpeg-q100" else None)
        ),
        "max_pixel_phase_spread": args.max_pixel_phase_spread,
        "sync_mode": args.sync_mode,
        "max_skew_us": args.max_skew_us,
        "strict_sync_us": args.strict_sync_us,
        "codec": args.codec,
        "webp_quality": args.webp_quality if args.codec == "webp" else None,
        "capture_strategy": args.capture_strategy,
        "manual_exposure_us": args.manual_exposure_us,
        "iso": args.iso,
        "manual_focus": args.manual_focus,
        "skip_frames": args.skip_frames,
        "camera_frame_selection": (
            "independent-streams-posthoc-nearest-device-timestamp-group"
            if args.capture_strategy == "stream-first"
            else "nearest-device-timestamp-group"
        ),
        "warmup_seconds": args.warmup_seconds,
        "image_workers_per_camera": args.image_workers_per_camera,
        "imu": {
            "enabled": args.mode in ("imu_calib", "vio"),
            "rate_hz": args.imu_rate,
            "batch_report_threshold": 10,
            "max_batch_reports": 20,
            "accelerometer": "ACCELEROMETER_RAW",
            "gyroscope": "GYROSCOPE_RAW",
            "model": imu_model,
            "range": "device-default",
        },
        "basalt_ready": False,
    }
    write_json_atomic(session_dir / "session.json", metadata)
    return session_dir, metadata


def send_exposure(queues, exposure_us: int, iso: int):
    for control_queue in queues.values():
        control = dai.CameraControl()
        control.setManualExposure(exposure_us, iso)
        control_queue.send(control)


def frame_to_gray(frame) -> np.ndarray:
    """Read direct GRAY8 or extract the NV12 luma plane without BGR conversion."""
    if isinstance(frame, dai.EncodedFrame) or not hasattr(frame, "getWidth"):
        encoded = np.asarray(frame.getData(), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError("无法解码设备端MJPEG预检帧")
        return image
    width = int(frame.getWidth())
    height = int(frame.getHeight())
    raw = frame.getFrame()
    if height <= 1 or (raw.ndim == 2 and raw.shape[0] == 1):
        image = cv2.imdecode(
            np.asarray(frame.getData(), dtype=np.uint8).reshape(-1),
            cv2.IMREAD_GRAYSCALE,
        )
        if image is None:
            raise RuntimeError("无法解码设备端MJPEG预检帧")
        return image
    if raw.ndim == 1 and raw.size >= width * height:
        raw = raw.reshape((-1, width))
    if raw.ndim == 2 and raw.shape[0] >= height and raw.shape[1] >= width:
        return np.ascontiguousarray(raw[:height, :width])
    decoded = frame.getCvFrame()
    if decoded.ndim == 2:
        return np.ascontiguousarray(decoded)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)


def preflight_camera_images(camera_queues, timeout_seconds: float = 3.0):
    """Collect one fresh image per camera before creating a session directory."""
    frames = {}
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline and len(frames) < len(CAMERA_IDS):
        for name, output_queue in camera_queues.items():
            messages = output_queue.tryGetAll()
            if messages:
                frames[name] = messages[-1]
        if len(frames) < len(CAMERA_IDS):
            time.sleep(0.001)
    missing = [name for name in CAMERA_IDS if name not in frames]
    if missing:
        raise RuntimeError(f"采集预检未收到相机帧：{missing}")
    return {name: frame_to_gray(frames[name]) for name in CAMERA_IDS}


def validate_preflight_images(images, max_phase_spread: float) -> dict[str, dict]:
    """Reject Bayer-like 2x2 phase structure before any dataset is created."""
    report = {}
    failures = []
    for name in CAMERA_IDS:
        image = images[name]
        if image.dtype != np.uint8 or image.ndim != 2:
            failures.append(f"{name}不是8-bit单通道图像：{image.dtype}/{image.shape}")
            continue
        if (image.shape[1], image.shape[0]) != SENSOR_RESOLUTION:
            failures.append(
                f"{name}分辨率为{image.shape[1]}x{image.shape[0]}，应为"
                f"{SENSOR_RESOLUTION[0]}x{SENSOR_RESOLUTION[1]}"
            )
            continue
        metrics = pixel_phase_metrics(image)
        report[name] = metrics
        if metrics["spread"] > max_phase_spread:
            failures.append(
                f"{name} 2x2像素相位差{metrics['spread']:.2f}>"
                f"{max_phase_spread:.2f}，疑似未去马赛克Bayer数据"
            )
    if failures:
        raise RuntimeError("；".join(failures))
    return report


def wait_for_aprilgrid(
    camera_queues,
    imu_queue,
    min_tags: int,
    timeout_seconds: float = 0.0,
    detector=None,
    stop_event: threading.Event | None = None,
) -> dict:
    """Drain live streams until any camera sees the configured AprilGrid."""
    detector = detector or create_detector()
    started = time.monotonic()
    last_status = started - 1.0
    best = {name: 0 for name in CAMERA_IDS}
    print(
        f"等待AprilGrid：任一路检测到至少{min_tags}个有效标签后开始采集...",
        flush=True,
    )
    while True:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("等待AprilGrid期间收到停止请求")
        for name, output_queue in camera_queues.items():
            messages = output_queue.tryGetAll()
            if not messages:
                continue
            image = frame_to_gray(messages[-1])
            _, tag_ids, _ = detect_aprilgrid(image, detector)
            best[name] = max(best[name], len(tag_ids))
            if len(tag_ids) >= min_tags:
                detected = {
                    "camera_id": name,
                    "tag_ids": sorted(tag_ids),
                    "detected_tags": len(tag_ids),
                    "wait_seconds": time.monotonic() - started,
                    "trigger_timestamp_ns": timedelta_ns(
                        messages[-1].getTimestampDevice()
                    ),
                }
                for queue_ in camera_queues.values():
                    queue_.tryGetAll()
                if imu_queue is not None:
                    imu_queue.tryGetAll()
                print(
                    f"已检测到AprilGrid：{name} tags={len(tag_ids)}；"
                    "从后续新帧开始记录。",
                    flush=True,
                )
                return detected
        if imu_queue is not None:
            imu_queue.tryGetAll()
        now = time.monotonic()
        if timeout_seconds > 0 and now - started >= timeout_seconds:
            raise RuntimeError(
                f"等待AprilGrid超过{timeout_seconds:.1f}秒；最佳检测={best}"
            )
        if now - last_status >= 1.0:
            print(
                "等待AprilGrid "
                + " ".join(f"{name}:{best[name]}" for name in CAMERA_IDS),
                flush=True,
            )
            last_status = now
        time.sleep(0.001)


def fit_preview(images, lines):
    tiles = []
    for name in CAMERA_IDS:
        image = images.get(name)
        if image is None:
            tile = np.zeros((200, 320, 3), dtype=np.uint8)
        else:
            tile = cv2.resize(image, (320, 200), interpolation=cv2.INTER_AREA)
            if tile.ndim == 2:
                tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        cv2.putText(tile, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        tiles.append(tile)
    body = cv2.vconcat((cv2.hconcat(tiles[:2]), cv2.hconcat(tiles[2:])))
    footer = np.full((72, body.shape[1], 3), 25, dtype=np.uint8)
    for index, line in enumerate(lines[:3]):
        cv2.putText(footer, line, (8, 21 + index * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    return cv2.vconcat((body, footer))


CaptureEventCallback = Callable[[str, dict], None]


def _emit(callback: CaptureEventCallback | None, event: str, **payload) -> None:
    """Publish optional, best-effort UI telemetry without affecting capture."""
    if callback is not None:
        try:
            callback(event, payload)
        except Exception:
            # A GUI observer must never terminate the hardware recording loop.
            pass


def run(
    args,
    event_callback: CaptureEventCallback | None = None,
    stop_event: threading.Event | None = None,
) -> Path:
    """Run one capture session.

    ``event_callback`` and ``stop_event`` form the headless capture-core API
    used by the Windows GUI.  Existing command-line callers retain identical
    behavior when both arguments are omitted.
    """
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        raise RuntimeError("未发现OAK设备")
    status, error = "completed", None
    session_dir = None
    writer = None
    with dai.Device(devices[0]) as device, dai.Pipeline(device) as pipeline:
        mapping = validate_camera_mapping(device.getConnectedCameras())
        _emit(
            event_callback,
            "device",
            device_id=device.getDeviceId(),
            camera_mapping={name: socket.name for name, socket in mapping.items()},
        )
        camera_queues, control_queues = {}, {}
        camera_host_queue_size = max(8, args.queue_size // len(CAMERA_IDS))
        for name in CAMERA_IDS:
            camera = pipeline.create(dai.node.Camera)
            # Keep COLOR to force the ISP/demosaic route even when stale device
            # metadata identifies these Bayer sensors as OV9282.
            camera.setSensorType(dai.CameraSensorType.COLOR)
            camera.build(mapping[name], sensorResolution=SENSOR_RESOLUTION, sensorFps=args.fps)
            if args.sync_mode == "fsin":
                camera.initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)
            if args.manual_exposure_us is not None:
                camera.initialControl.setManualExposure(args.manual_exposure_us, args.iso)
            if args.manual_focus is not None:
                camera.initialControl.setManualFocus(args.manual_focus)
            transfer_type = (
                dai.ImgFrame.Type.GRAY8
                if args.camera_transfer_format == "gray8"
                else dai.ImgFrame.Type.NV12
            )
            output = camera.requestFullResolutionOutput(type=transfer_type, fps=args.fps)
            if args.camera_transfer_format.startswith("mjpeg-"):
                encoder = pipeline.create(dai.node.VideoEncoder)
                encoder.setDefaultProfilePreset(
                    args.fps, dai.VideoEncoderProperties.Profile.MJPEG
                )
                encoder.setLossless(
                    args.camera_transfer_format == "mjpeg-lossless"
                )
                encoder.setQuality(
                    95 if args.camera_transfer_format == "mjpeg-q95" else 100
                )
                output.link(encoder.input)
                camera_queues[name] = encoder.bitstream.createOutputQueue(
                    maxSize=camera_host_queue_size, blocking=False
                )
            else:
                camera_queues[name] = output.createOutputQueue(
                    maxSize=camera_host_queue_size, blocking=False
                )
            control_queues[name] = camera.inputControl.createInputQueue(maxSize=4, blocking=False)

        imu_queue = None
        if args.mode in ("imu_calib", "vio"):
            imu = pipeline.create(dai.node.IMU)
            imu.enableIMUSensor(
                [dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW],
                args.imu_rate,
            )
            # Preserve every device-timestamped sample while reducing XLink
            # transaction overhead. At 400 Hz a batch of 10 adds only ~25 ms
            # delivery latency, which has no effect on offline synchronization.
            imu.setBatchReportThreshold(10)
            imu.setMaxBatchReports(20)
            imu_queue = imu.out.createOutputQueue(
                maxSize=args.queue_size, blocking=False
            )

        pipeline.start()
        usb_speed = str(device.getUsbSpeed())
        _emit(event_callback, "usb_speed", usb_speed=usb_speed)
        if args.require_usb3 and "SUPER" not in usb_speed.upper():
            raise RuntimeError(
                f"运行态OAK链路为{usb_speed}，严格采集要求USB 3 SuperSpeed；"
                "请更换USB3端口/线缆后重试"
            )
        _emit(event_callback, "waiting_fsin", sync_mode=args.sync_mode)
        warmup_deadline = time.monotonic() + args.warmup_seconds
        while pipeline.isRunning() and time.monotonic() < warmup_deadline:
            for output_queue in camera_queues.values():
                output_queue.tryGetAll()
            if imu_queue is not None:
                imu_queue.tryGetAll()
            time.sleep(0.001)
        preflight_images = preflight_camera_images(camera_queues)
        preflight_quality = validate_preflight_images(
            preflight_images, args.max_pixel_phase_spread
        )
        _emit(event_callback, "preflight_passed", pixel_phase=preflight_quality)
        start_gate = None
        if args.start_on_aprilgrid:
            start_gate = wait_for_aprilgrid(
                camera_queues,
                imu_queue,
                args.start_min_tags,
                args.start_timeout,
                stop_event=stop_event,
            )
            _emit(event_callback, "aprilgrid_detected", **start_gate)
        raw_sensor_names = device.getCameraSensorNames()
        camera_sensor_names = {
            getattr(socket, "name", str(socket)): str(sensor)
            for socket, sensor in raw_sensor_names.items()
        }
        imu_model = str(device.getConnectedIMU()) if args.mode in ("imu_calib", "vio") else None
        session_dir, metadata = create_session(
            args,
            device.getDeviceId(),
            dai.__version__,
            camera_sensor_names,
            imu_model,
        )
        metadata["device_host_queue_size"] = {
            "camera_per_stream": camera_host_queue_size,
            "imu": args.queue_size if imu_queue is not None else 0,
        }
        metadata["runtime_usb_speed"] = usb_speed
        metadata["preflight_pixel_phase"] = preflight_quality
        metadata["start_gate"] = {
            "mode": "first-aprilgrid" if args.start_on_aprilgrid else "immediate",
            "min_tags": args.start_min_tags if args.start_on_aprilgrid else None,
            "detection": start_gate,
        }
        write_json_atomic(session_dir / "session.json", metadata)
        writer = DatasetWriter(
            session_dir,
            args.codec,
            args.webp_quality,
            args.queue_size,
            args.image_workers_per_camera,
        )
        writer.start()
        _emit(event_callback, "recording", session_dir=str(session_dir))
        sync_buffer_size = max(8, min(16, camera_host_queue_size))
        camera_states = {
            name: CameraFrameState(sync_buffer_size) for name in CAMERA_IDS
        }
        matcher = TimestampGroupMatcher(args.max_skew_us)
        previews = {}
        seen = {name: 0 for name in CAMERA_IDS}
        started = time.monotonic()
        last_dashboard = started
        last_counts = seen.copy()
        last_matched_groups = 0
        imu_seen = {"accel": 0, "gyro": 0}
        last_imu_counts = imu_seen.copy()
        latest_exposure = {name: 0 for name in CAMERA_IDS}
        matched_groups = 0
        submitted_groups = 0
        last_group_skew_us = None
        max_live_group_skew_us = None
        sweep_index = 0
        last_preview_emit = started
        receiver_stop = threading.Event()
        receiver_threads: list[threading.Thread] = []
        receiver_errors: list[BaseException] = []
        receiver_error_lock = threading.Lock()

        def remember_receiver_error(exc: BaseException) -> None:
            with receiver_error_lock:
                if not receiver_errors:
                    receiver_errors.append(exc)
            receiver_stop.set()

        def receive_camera(name: str) -> None:
            try:
                output_queue = camera_queues[name]
                while pipeline.isRunning() and not receiver_stop.is_set():
                    messages = output_queue.tryGetAll()
                    if not messages:
                        receiver_stop.wait(0.001)
                        continue
                    for frame in messages:
                        if receiver_stop.is_set():
                            break
                        seen[name] += 1
                        sample = camera_states[name].consume(
                            frame, retain_for_matching=False
                        )
                        latest_exposure[name] = sample.exposure_us
                        writer.submit(
                            ImageTask(
                                name,
                                sample.timestamp_ns,
                                sample.sequence_num,
                                sample.exposure_us,
                                sample.iso,
                                sample.frame,
                            )
                        )
            except BaseException as exc:
                remember_receiver_error(exc)

        def receive_imu() -> None:
            try:
                while pipeline.isRunning() and not receiver_stop.is_set():
                    messages = imu_queue.tryGetAll()
                    if not messages:
                        receiver_stop.wait(0.001)
                        continue
                    for data in messages:
                        for packet in data.packets:
                            for sensor, report in (
                                ("accel", packet.acceleroMeter),
                                ("gyro", packet.gyroscope),
                            ):
                                imu_seen[sensor] += 1
                                writer.submit(
                                    ImuTask(
                                        sensor,
                                        timedelta_ns(report.getTimestampDevice()),
                                        int(data.getSequenceNum()),
                                        float(report.x),
                                        float(report.y),
                                        float(report.z),
                                    )
                                )
            except BaseException as exc:
                remember_receiver_error(exc)

        if args.capture_strategy == "stream-first":
            receiver_threads = [
                threading.Thread(
                    target=receive_camera,
                    args=(name,),
                    name=f"oak4p-receiver-{name}",
                    daemon=True,
                )
                for name in CAMERA_IDS
            ]
            if imu_queue is not None:
                receiver_threads.append(
                    threading.Thread(
                        target=receive_imu,
                        name="oak4p-receiver-imu",
                        daemon=True,
                    )
                )
            for thread in receiver_threads:
                thread.start()
        try:
            while pipeline.isRunning():
                now = time.monotonic()
                if stop_event is not None and stop_event.is_set():
                    status = "interrupted"
                    break
                if args.capture_strategy == "stream-first":
                    if receiver_errors:
                        raise RuntimeError(
                            f"采集接收线程失败：{receiver_errors[0]}"
                        ) from receiver_errors[0]
                    if writer.error:
                        raise RuntimeError(f"写盘线程失败：{writer.error}")
                    if now - last_dashboard >= 1.0:
                        elapsed = now - last_dashboard
                        camera_rates = {
                            name: (seen[name] - last_counts[name]) / elapsed
                            for name in CAMERA_IDS
                        }
                        imu_rates = {
                            name: (imu_seen[name] - last_imu_counts[name]) / elapsed
                            for name in imu_seen
                        }
                        line = (
                            f"fps={','.join(f'{name}:{camera_rates[name]:.1f}' for name in CAMERA_IDS)} "
                            f"imu={imu_rates['accel']:.0f}/{imu_rates['gyro']:.0f}Hz "
                            f"queue={writer.queue_depth}/{writer.queue_capacity} "
                            "sync=posthoc "
                            f"dropped={sum(writer.dropped.values())}"
                        )
                        print(line, flush=True)
                        _emit(
                            event_callback,
                            "statistics",
                            camera_fps=camera_rates,
                            imu_hz=imu_rates,
                            queue_depth=writer.queue_depth,
                            queue_capacity=writer.queue_capacity,
                            group_rate=None,
                            skew_us=None,
                            unmatched=None,
                            dropped=sum(writer.dropped.values()),
                        )
                        last_counts = seen.copy()
                        last_imu_counts = imu_seen.copy()
                        last_dashboard = now
                    if args.duration is not None and now - started >= args.duration:
                        break
                    time.sleep(0.001)
                    continue
                camera_batches = {
                    name: output_queue.tryGetAll()
                    for name, output_queue in camera_queues.items()
                }
                batch_length = max(
                    (len(messages) for messages in camera_batches.values()),
                    default=0,
                )
                for batch_index in range(batch_length):
                    for name in CAMERA_IDS:
                        messages = camera_batches[name]
                        if batch_index >= len(messages):
                            continue
                        frame = messages[batch_index]
                        seen[name] += 1
                        sample = camera_states[name].consume(frame)
                        latest_exposure[name] = sample.exposure_us
                    while True:
                        group = matcher.match(camera_states)
                        if group is None:
                            break
                        matched_groups += 1
                        last_group_skew_us = group_skew_us(group)
                        max_live_group_skew_us = (
                            last_group_skew_us
                            if max_live_group_skew_us is None
                            else max(max_live_group_skew_us, last_group_skew_us)
                        )
                        if not should_record_group(
                            matched_groups, args.skip_frames
                        ):
                            continue
                        preview_images = {
                            name: frame_to_gray(group[name].frame).copy()
                            for name in CAMERA_IDS
                        }
                        previews.update(preview_images)
                        if now - last_preview_emit >= 0.2:
                            _emit(event_callback, "preview", images=preview_images)
                            last_preview_emit = now
                        tasks = {
                            name: ImageTask(
                                name,
                                group[name].timestamp_ns,
                                group[name].sequence_num,
                                group[name].exposure_us,
                                group[name].iso,
                                (
                                    group[name].frame
                                    if args.codec == "nv12"
                                    else preview_images[name]
                                ),
                            )
                            for name in CAMERA_IDS
                        }
                        if writer.submit_camera_group(tasks):
                            submitted_groups += 1
                if imu_queue is not None:
                    for data in imu_queue.tryGetAll():
                        for packet in data.packets:
                            for sensor, report in (
                                ("accel", packet.acceleroMeter),
                                ("gyro", packet.gyroscope),
                            ):
                                imu_seen[sensor] += 1
                                writer.submit(
                                    ImuTask(
                                        sensor,
                                        timedelta_ns(report.getTimestampDevice()),
                                        int(data.getSequenceNum()),
                                        float(report.x),
                                        float(report.y),
                                        float(report.z),
                                    )
                                )
                if args.mode == "response":
                    target_index = min(
                        len(args.exposure_sweep_us) - 1,
                        seen[CAMERA_IDS[0]] // args.frames_per_exposure,
                    )
                    if target_index != sweep_index:
                        sweep_index = target_index
                        send_exposure(control_queues, args.exposure_sweep_us[sweep_index], args.iso)
                if writer.error:
                    raise RuntimeError(f"写盘线程失败：{writer.error}")
                if now - last_dashboard >= 1.0:
                    elapsed = now - last_dashboard
                    camera_rates = {
                        name: (seen[name] - last_counts[name]) / elapsed for name in CAMERA_IDS
                    }
                    imu_rates = {
                        name: (imu_seen[name] - last_imu_counts[name]) / elapsed
                        for name in imu_seen
                    }
                    group_rate = (matched_groups - last_matched_groups) / elapsed
                    skew_text = (
                        "waiting"
                        if last_group_skew_us is None
                        else f"{last_group_skew_us:.0f}us"
                    )
                    line = (
                        f"fps={','.join(f'{name}:{camera_rates[name]:.1f}' for name in CAMERA_IDS)} "
                        f"imu={imu_rates['accel']:.0f}/{imu_rates['gyro']:.0f}Hz "
                        f"queue={writer.queue_depth}/{writer.queue_capacity} "
                        f"sync={group_rate:.1f}grp/s skew={skew_text} "
                        f"unmatched={matcher.unmatched_references} "
                        f"dropped={sum(writer.dropped.values())}"
                    )
                    print(line, flush=True)
                    _emit(
                        event_callback,
                        "statistics",
                        camera_fps=camera_rates,
                        imu_hz=imu_rates,
                        queue_depth=writer.queue_depth,
                        queue_capacity=writer.queue_capacity,
                        group_rate=group_rate,
                        skew_us=last_group_skew_us,
                        unmatched=matcher.unmatched_references,
                        dropped=sum(writer.dropped.values()),
                    )
                    last_counts = seen.copy()
                    last_imu_counts = imu_seen.copy()
                    last_matched_groups = matched_groups
                    last_dashboard = now
                if not args.headless:
                    skew_text = (
                        "waiting"
                        if last_group_skew_us is None
                        else f"{last_group_skew_us:.0f}us"
                    )
                    view = fit_preview(
                        previews,
                        (
                            f"mode={args.mode} sync={args.sync_mode} codec={args.codec}",
                            f"groups={matched_groups} unmatched={matcher.unmatched_references} skew={skew_text}",
                            f"exposure={latest_exposure} Q/ESC=stop",
                        ),
                    )
                    cv2.imshow(WINDOW_NAME, view)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                if args.duration is not None and now - started >= args.duration:
                    break
                time.sleep(0.001)
        except KeyboardInterrupt:
            status = "interrupted"
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            receiver_stop.set()
            for thread in receiver_threads:
                thread.join(timeout=5.0)
            pipeline.stop()
            for thread in receiver_threads:
                thread.join(timeout=1.0)
            lingering_receivers = [
                thread.name for thread in receiver_threads if thread.is_alive()
            ]
            if lingering_receivers:
                status = "error"
                error = error or f"采集接收线程未停止：{lingering_receivers}"
            cv2.destroyAllWindows()
            if writer is not None:
                try:
                    writer.close()
                except Exception as exc:
                    status = "error"
                    error = error or f"{type(exc).__name__}: {exc}"
                    if sys.exc_info()[0] is None:
                        raise
            if session_dir is not None:
                rows = {name: load_camera_rows(session_dir, name) for name in CAMERA_IDS}
                groups = build_groups(rows, args.max_skew_us)
                write_groups(session_dir / "groups.csv", groups)
                skew_values = [int(group["skew_us"]) for group in groups]
                if args.capture_strategy == "stream-first":
                    matched_groups = len(groups)
                    submitted_groups = len(groups)
                    unmatched_references = max(
                        0, len(rows[CAMERA_IDS[0]]) - len(groups)
                    )
                    last_unmatched_skew_us = None
                    max_live_group_skew_us = (
                        max(skew_values) if skew_values else None
                    )
                else:
                    unmatched_references = matcher.unmatched_references
                    last_unmatched_skew_us = matcher.last_unmatched_skew_us
                metadata.update(
                    {
                        "status": status,
                        "ended_at": datetime.now().astimezone().isoformat(timespec="microseconds"),
                        "error": error,
                        "written": writer.written,
                        "dropped": writer.dropped,
                        "writer_queue_high_watermark": writer.high_watermark,
                        "writer_queue_high_watermark_by_queue": writer.high_watermark_by_queue,
                        "camera_host_dropped": {
                            name: camera_states[name].host_dropped
                            for name in CAMERA_IDS
                        },
                        "camera_sync_buffer_evicted": {
                            name: camera_states[name].buffer_evicted
                            for name in CAMERA_IDS
                        },
                        "sync_matcher": {
                            "buffer_size": sync_buffer_size,
                            "matched_groups": matched_groups,
                            "submitted_groups": submitted_groups,
                            "unmatched_references": unmatched_references,
                            "last_unmatched_skew_us": last_unmatched_skew_us,
                            "max_live_group_skew_us": max_live_group_skew_us,
                        },
                        "groups": len(groups),
                        "max_observed_group_skew_us": max(skew_values) if skew_values else None,
                        "basalt_ready": False,
                    }
                )
                write_json_atomic(session_dir / "session.json", metadata)
                _emit(event_callback, "finished", session_dir=str(session_dir), status=status)
    return session_dir


def main(argv=None):
    args = parse_args(argv)
    try:
        session = run(args)
    except (RuntimeError, OSError, cv2.error) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    print(f"数据目录：{session}")
    print(f"下一步：python -m dohc2_c4lib.validate_oak4p_dataset {session}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
