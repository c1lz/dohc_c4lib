#!/usr/bin/env python3
"""实时验证 OAK-4P 四相机外部 FSIN 硬同步。

四路帧按 ``getTimestampDevice`` 的最近邻时间戳在线配组，再计算组内偏差。
OAK 的各相机序列号独立计数，不能跨相机作为同一触发帧的依据；序列号仅用于
统计主机输出队列丢帧。此方式不会把主机读取时序、显示延迟或独立序号偏移误判为
相机不同步。若某一路缺少对应帧，相邻触发帧不会被用于计算 device skew，而会
单独计为 unmatched。
按 q、Esc 或 Ctrl+C 退出。
"""

from __future__ import annotations

import sys
import argparse
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path

if Path("/usr/share/fonts/truetype/dejavu").is_dir():
    import os

    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import depthai as dai
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dohc2_c4lib import show_oak4p_fsin_sync as base


WINDOW_NAME = "OAK-4P FSIN Hardware Sync Verification"
DEFAULT_PREVIEW_SIZE = (640, 400)


@dataclass(frozen=True)
class FrameSample:
    sequence_num: int
    timestamp_ns: int
    image: np.ndarray


class SequenceRateMeter:
    def __init__(self, window_seconds: float = 1.0):
        self.window_ns = int(window_seconds * 1e9)
        self.samples: deque[tuple[int, int]] = deque()

    def update(self, timestamp_ns: int, sequence_num: int) -> None:
        self.samples.append((timestamp_ns, sequence_num))
        cutoff = timestamp_ns - self.window_ns
        while len(self.samples) > 2 and self.samples[0][0] < cutoff:
            self.samples.popleft()

    @property
    def value(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        first_timestamp, first_sequence = self.samples[0]
        last_timestamp, last_sequence = self.samples[-1]
        elapsed_s = (last_timestamp - first_timestamp) / 1e9
        sequence_delta = last_sequence - first_sequence
        if elapsed_s <= 0 or sequence_delta <= 0:
            return 0.0
        return sequence_delta / elapsed_s


class CameraState:
    """维护一路最新帧及按序列号等待配组的有限缓存。"""

    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.pending: OrderedDict[int, FrameSample] = OrderedDict()
        self.last_sequence: int | None = None
        self.host_dropped = 0
        self.rate = SequenceRateMeter()

    def consume(self, frame) -> FrameSample:
        sequence_num = int(frame.getSequenceNum())
        if self.last_sequence is not None and sequence_num > self.last_sequence + 1:
            self.host_dropped += sequence_num - self.last_sequence - 1
        self.last_sequence = sequence_num
        sample = FrameSample(
            sequence_num=sequence_num,
            timestamp_ns=timedelta_ns(frame.getTimestampDevice()),
            image=frame.getCvFrame(),
        )
        self.pending[sequence_num] = sample
        self.rate.update(sample.timestamp_ns, sequence_num)
        while len(self.pending) > self.buffer_size:
            self.pending.popitem(last=False)
        return sample


class SkewStatistics:
    def __init__(self, window_seconds: float):
        self.window_seconds = window_seconds
        self.samples: deque[tuple[float, float]] = deque()

    def update(self, skew_us: float, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self.samples.append((now, skew_us))
        cutoff = now - self.window_seconds
        while len(self.samples) > 1 and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def summary(self) -> dict[str, float] | None:
        if not self.samples:
            return None
        values = np.asarray([sample[1] for sample in self.samples], dtype=float)
        return {
            "p50_us": float(np.percentile(values, 50)),
            "p95_us": float(np.percentile(values, 95)),
            "p99_us": float(np.percentile(values, 99)),
            "max_us": float(np.max(values)),
            "groups": float(len(values)),
        }


def validate_camera_mapping(sockets) -> dict[str, object]:
    by_name = {socket.name: socket for socket in sockets}
    missing = [socket for socket in CAMERA_MAPPING.values() if socket not in by_name]
    if missing:
        raise RuntimeError(f"缺少相机接口：{missing}；实际为{sorted(by_name)}")
    return {name: by_name[CAMERA_MAPPING[name]] for name in CAMERA_IDS}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="实时显示 OAK-4P 四相机与真实硬同步偏差")
    parser.add_argument("--sync-mode", choices=("fsin", "free-run"), default="fsin")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--preview-width", type=int, default=DEFAULT_PREVIEW_SIZE[0])
    parser.add_argument("--preview-height", type=int, default=DEFAULT_PREVIEW_SIZE[1])
    parser.add_argument("--strict-sync-us", type=float, default=1_000.0)
    parser.add_argument(
        "--pairing-window-us",
        type=float,
        help="同一触发帧的最大关联窗口；默认0.45个帧周期",
    )
    parser.add_argument("--stats-window-seconds", type=float, default=5.0)
    parser.add_argument("--queue-size", type=int, default=16)
    args = parser.parse_args(argv)
    if args.fps <= 0 or args.strict_sync_us <= 0 or args.stats_window_seconds <= 0:
        parser.error("fps、strict-sync-us和stats-window-seconds必须大于0")
    if args.pairing_window_us is None:
        args.pairing_window_us = 0.45 * 1_000_000.0 / args.fps
    if args.pairing_window_us <= 0:
        parser.error("pairing-window-us必须大于0")
    if args.pairing_window_us >= 0.5 * 1_000_000.0 / args.fps:
        parser.error("pairing-window-us必须小于半个帧周期，以免关联相邻触发帧")
    if args.preview_width <= 0 or args.preview_height <= 0:
        parser.error("预览尺寸必须大于0")
    if args.queue_size < 2:
        parser.error("queue-size必须>=2")
    return args


def text(canvas, value: str, position, color, scale=0.55, thickness=1):
    cv2.putText(
        canvas,
        value,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def make_tile(name: str, sample: FrameSample, state: CameraState, tile_size, delta_us: float) -> np.ndarray:
    width, height = tile_size
    image = sample.image
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    tile = np.zeros((height, width, 3), dtype=np.uint8)
    left = (width - resized.shape[1]) // 2
    top = (height - resized.shape[0]) // 2
    tile[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    text(tile, f"{name}/{CAMERA_MAPPING[name]}  {state.rate.value:5.1f} FPS", (10, 26), (70, 240, 70), 0.55, 2)
    text(tile, f"seq={sample.sequence_num}  host-drop={state.host_dropped}", (10, 51), (230, 230, 230), 0.48)
    text(tile, f"device={sample.timestamp_ns / 1e9:.6f} s", (10, 74), (230, 230, 230), 0.48)
    text(tile, f"delta vs cam0={delta_us:+.1f} us", (10, 97), (230, 230, 230), 0.48)
    return tile


def build_view(
    group,
    states,
    args,
    statistics,
    matcher,
    usb_speed,
    tile_renderer=None,
) -> np.ndarray:
    tile_renderer = make_tile if tile_renderer is None else tile_renderer
    reference_timestamp_ns = group["cam0"].timestamp_ns
    tiles = [
        tile_renderer(
            name,
            group[name],
            states[name],
            (args.preview_width, args.preview_height),
            (group[name].timestamp_ns - reference_timestamp_ns) / 1_000.0,
        )
        for name in CAMERA_IDS
    ]
    body = cv2.vconcat((cv2.hconcat(tiles[:2]), cv2.hconcat(tiles[2:])))
    skew = group_skew_us(group)
    passed = skew <= args.strict_sync_us
    summary = statistics.summary()
    footer = np.full((106, body.shape[1], 3), 24, dtype=np.uint8)
    color = (70, 230, 70) if passed else (30, 30, 255)
    text(footer, f"{args.sync_mode.upper()}  cam0-seq={group['cam0'].sequence_num}  device skew={skew:.1f} us  limit={args.strict_sync_us:.1f} us  {'PASS' if passed else 'FAIL'}", (10, 30), color, 0.60, 2)
    if summary:
        text(footer, f"last {args.stats_window_seconds:.0f}s: groups={int(summary['groups'])}  p50={summary['p50_us']:.1f} us  p95={summary['p95_us']:.1f} us  p99={summary['p99_us']:.1f} us  max={summary['max_us']:.1f} us", (10, 58), (230, 230, 230), 0.48)
    unmatched = (
        "none"
        if matcher.last_unmatched_skew_us is None
        else f"{matcher.last_unmatched_skew_us:.1f} us"
    )
    text(
        footer,
        f"pair-window={args.pairing_window_us:.0f} us  unmatched={matcher.unmatched_references}  last-nearest={unmatched}  USB={usb_speed}; q/Esc exits",
        (10, 87),
        (220, 220, 100),
        0.48,
    )
    return cv2.vconcat((body, footer))


def waiting_view(args, states, matcher) -> np.ndarray:
    image = np.zeros((args.preview_height * 2 + 106, args.preview_width * 2, 3), dtype=np.uint8)
    waiting = [name for name in CAMERA_IDS if states[name].last_sequence is None]
    text(image, "Waiting for four camera frames...", (30, image.shape[0] // 2 - 18), (0, 230, 255), 0.8, 2)
    text(image, f"sync={args.sync_mode}; waiting={','.join(waiting) if waiting else 'matching device timestamps'}", (30, image.shape[0] // 2 + 18), (230, 230, 230), 0.56)
    text(image, f"unmatched={matcher.unmatched_references}; pairing-window={args.pairing_window_us:.0f} us", (30, image.shape[0] // 2 + 50), (230, 230, 230), 0.56)
    return image


def main(argv=None, *, tile_renderer=None, window_name=WINDOW_NAME) -> int:
    args = parse_args(argv)
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        raise RuntimeError("未发现OAK设备；请检查供电和USB连接")
    with dai.Device(devices[0]) as device, dai.Pipeline(device) as pipeline:
        mapping = validate_camera_mapping(device.getConnectedCameras())
        queues = {}
        for name in CAMERA_IDS:
            camera = pipeline.create(dai.node.Camera)
            camera.setSensorType(dai.CameraSensorType.MONO)
            camera.build(mapping[name], sensorResolution=SENSOR_RESOLUTION, sensorFps=args.fps)
            if args.sync_mode == "fsin":
                camera.initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)
            output = camera.requestOutput(
                (args.preview_width, args.preview_height),
                type=dai.ImgFrame.Type.GRAY8,
                resizeMode=dai.ImgResizeMode.LETTERBOX,
                fps=args.fps,
            )
            queues[name] = output.createOutputQueue(maxSize=args.queue_size, blocking=False)

        pipeline.start()
        usb_speed = device.getUsbSpeed().name
        states = {name: CameraState(args.queue_size * 2) for name in CAMERA_IDS}
        matcher = TimestampGroupMatcher(args.pairing_window_us)
        statistics = SkewStatistics(args.stats_window_seconds)
        print(f"设备={device.getDeviceId()} USB={usb_speed} 映射={CAMERA_MAPPING}")
        print(f"同步模式={args.sync_mode}，严格门槛={args.strict_sync_us:g} us；按 q/Esc 退出。")
        if args.sync_mode == "fsin":
            print("FSIN模式要求外部触发持续输入；无触发时相机不会出帧。")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, args.preview_width * 2, args.preview_height * 2 + 106)
        display_group = None
        try:
            while pipeline.isRunning():
                for name, queue in queues.items():
                    for frame in queue.tryGetAll():
                        if not isinstance(frame, dai.ImgFrame):
                            raise RuntimeError(f"{name}返回意外消息类型：{type(frame)}")
                        states[name].consume(frame)
                group = matcher.match(states)
                if group:
                    skew = group_skew_us(group)
                    statistics.update(skew)
                    display_group = group
                if display_group:
                    cv2.imshow(
                        window_name,
                        build_view(
                            display_group,
                            states,
                            args,
                            statistics,
                            matcher,
                            usb_speed,
                            tile_renderer,
                        ),
                    )
                else:
                    cv2.imshow(window_name, waiting_view(args, states, matcher))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
        except KeyboardInterrupt:
            print("\n已停止。")
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
