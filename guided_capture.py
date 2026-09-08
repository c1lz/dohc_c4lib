#!/usr/bin/env python3
"""Guided one- or two-camera AprilGrid collection core."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

if Path("/usr/share/fonts/truetype/dejavu").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import depthai as dai
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dohc2_c4lib.aprilgrid_common import (
    CAMERA_IDS,
    CAMERA_MAPPING,
    SENSOR_RESOLUTION,
    ImageAssessment,
    QualityThresholds,
    create_detector,
    image_metrics,
    quality_reasons,
    timestamp_us,
    update_stability,
    write_json_atomic,
)


WINDOW_NAME = "OAK-4P Guided AprilGrid Collector"
DISPLAY_SIZE = (960, 600)
GRID_COLORS = ((70, 70, 70), (60, 190, 60), (0, 190, 255))
POSITION_NAMES = (
    "upper-left", "upper-center", "upper-right",
    "center-left", "center", "center-right",
    "lower-left", "lower-center", "lower-right",
)
# Center and corners first so the minimum six bins are spatially distributed.
POSITION_GUIDANCE_ORDER = (4, 0, 2, 6, 8, 1, 7, 3, 5)
TILT_TARGETS = (
    "front",
    "yaw-left",
    "yaw-right",
    "pitch-up",
    "pitch-down",
    "roll-left",
    "roll-right",
)
SCALE_TARGETS = ("far", "mid", "near")


@dataclass
class GuidedAssessment:
    base: ImageAssessment
    annotated: np.ndarray
    area_ratio: float
    position_bin: int | None
    scale_class: str | None
    tilt_class: str | None
    pose_signature: tuple | None


@dataclass
class StreamState:
    frame: object | None = None
    preview: np.ndarray | None = None
    assessment: GuidedAssessment | None = None
    last_detection_at: float = float("-inf")
    fps_samples: deque = None

    def __post_init__(self):
        self.fps_samples = deque()

    def update_rate(self, stamp_us):
        stamp = stamp_us / 1_000_000.0
        self.fps_samples.append(stamp)
        while len(self.fps_samples) > 2 and self.fps_samples[0] < stamp - 1.0:
            self.fps_samples.popleft()

    @property
    def fps(self):
        if len(self.fps_samples) < 2:
            return 0.0
        elapsed = self.fps_samples[-1] - self.fps_samples[0]
        return (len(self.fps_samples) - 1) / elapsed if elapsed > 0 else 0.0


def add_common_arguments(parser):
    parser.add_argument("--output-dir", type=Path, default=Path("guided_datasets"))
    parser.add_argument("--sync-mode", choices=("free-run", "fsin"), default="free-run")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--detection-fps", type=float, default=4.0)
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    parser.add_argument("--min-tags", type=int, default=6)
    parser.add_argument("--min-sharpness", type=float, default=80.0)
    parser.add_argument("--min-brightness", type=float, default=45.0)
    parser.add_argument("--max-brightness", type=float, default=220.0)
    parser.add_argument("--max-clipped-percent", type=float, default=30.0)
    parser.add_argument("--max-motion-px", type=float, default=4.0)
    parser.add_argument("--min-area-ratio", type=float, default=0.025)
    parser.add_argument("--max-area-ratio", type=float, default=0.65)
    parser.add_argument("--max-skew-us", type=int)
    parser.add_argument(
        "--manual-focus",
        type=int,
        help="可选固定对焦位置（0..255）；自动对焦模组建议全程锁定",
    )


def validate_args(parser, args):
    if args.fps <= 0 or args.detection_fps <= 0:
        parser.error("--fps和--detection-fps必须大于0")
    if args.settle_seconds < 0:
        parser.error("--settle-seconds不能小于0")
    if not 1 <= args.min_tags <= 36:
        parser.error("--min-tags必须在1到36之间")
    if not 0 <= args.min_brightness <= args.max_brightness <= 255:
        parser.error("亮度阈值必须满足0<=min<=max<=255")
    if not 0 <= args.max_clipped_percent <= 100:
        parser.error("--max-clipped-percent必须在0到100之间")
    if args.min_sharpness < 0 or args.max_motion_px < 0:
        parser.error("清晰度和运动阈值不能小于0")
    if not 0 < args.min_area_ratio < args.max_area_ratio <= 1:
        parser.error("面积比例必须满足0<min<max<=1")
    if args.max_skew_us is None:
        args.max_skew_us = 1_000 if args.sync_mode == "fsin" else 20_000
    if args.max_skew_us <= 0:
        parser.error("--max-skew-us必须大于0")
    if args.manual_focus is not None and not 0 <= args.manual_focus <= 255:
        parser.error("--manual-focus必须在0到255之间")
    return args


def thresholds_from_args(args):
    return QualityThresholds(
        min_tags=args.min_tags,
        min_sharpness=args.min_sharpness,
        min_brightness=args.min_brightness,
        max_brightness=args.max_brightness,
        max_clipped_percent=args.max_clipped_percent,
    )


def filtered_detections(image, detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    result = []
    seen = set()
    if ids is not None:
        for corner, marker_id in zip(corners, ids.reshape(-1)):
            marker_id = int(marker_id)
            if 0 <= marker_id < 36 and marker_id not in seen:
                seen.add(marker_id)
                result.append((marker_id, np.asarray(corner).reshape(4, 2)))
    return result


def target_corner(marker_id, corner_index):
    row, column = divmod(marker_id, 6)
    pitch = 1.3
    x = column * pitch
    y = row * pitch
    offsets = ((0, 0), (1, 0), (1, 1), (0, 1))
    dx, dy = offsets[corner_index]
    return x + dx, y + dy


def projected_board(detections):
    if len(detections) < 2:
        return None
    object_points = []
    image_points = []
    for marker_id, corners in detections:
        for index, corner in enumerate(corners):
            object_points.append(target_corner(marker_id, index))
            image_points.append(corner)
    homography, _ = cv2.findHomography(
        np.asarray(object_points, dtype=np.float32),
        np.asarray(image_points, dtype=np.float32),
        cv2.RANSAC,
        3.0,
    )
    if homography is None:
        return None
    extent = 5 * 1.3 + 1
    outer = np.asarray(
        [[[0, 0], [extent, 0], [extent, extent], [0, extent]]],
        dtype=np.float32,
    )
    return cv2.perspectiveTransform(outer, homography).reshape(4, 2)


def geometry_features(quad, width, height):
    if quad is None:
        return 0.0, None, 0.0, 0.0, 0.0, None
    area_ratio = abs(cv2.contourArea(quad.astype(np.float32))) / (width * height)
    center = np.mean(quad, axis=0)
    column = min(2, max(0, int(center[0] * 3 / width)))
    row = min(2, max(0, int(center[1] * 3 / height)))
    position_bin = row * 3 + column
    top = np.linalg.norm(quad[1] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    yaw = (left - right) / max(left + right, 1e-6)
    pitch = (top - bottom) / max(top + bottom, 1e-6)
    top_vector = quad[1] - quad[0]
    roll_degrees = math.degrees(math.atan2(top_vector[1], top_vector[0]))
    signature = (
        float(center[0] / width),
        float(center[1] / height),
        float(area_ratio),
        float(yaw),
        float(pitch),
        float(roll_degrees),
    )
    return area_ratio, position_bin, yaw, pitch, roll_degrees, signature


def classify_geometry(quad, width, height):
    area_ratio, position_bin, yaw, pitch, roll_degrees, _ = geometry_features(
        quad, width, height
    )
    if position_bin is None:
        return 0.0, None, None, None
    scale = "far" if area_ratio < 0.08 else "mid" if area_ratio < 0.23 else "near"
    if abs(roll_degrees) >= 12.0:
        tilt = "roll-right" if roll_degrees > 0 else "roll-left"
    elif max(abs(yaw), abs(pitch)) < 0.10:
        tilt = "front"
    elif abs(yaw) >= abs(pitch):
        tilt = "yaw-left" if yaw > 0 else "yaw-right"
    else:
        tilt = "pitch-up" if pitch > 0 else "pitch-down"
    return area_ratio, position_bin, scale, tilt


def pose_signatures_duplicate(current, saved):
    """Compare continuous board geometry instead of coarse category equality."""
    if not current or not saved or set(current) != set(saved):
        return False
    for name, signature in current.items():
        previous = saved.get(name)
        if signature is None or previous is None or len(previous) != 6:
            return False
        cx, cy, area, yaw, pitch, roll = map(float, signature)
        px, py, old_area, old_yaw, old_pitch, old_roll = map(float, previous)
        if math.hypot(cx - px, cy - py) >= 0.06:
            return False
        if min(area, old_area) <= 0 or max(area, old_area) / min(area, old_area) >= 1.25:
            return False
        if abs(yaw - old_yaw) >= 0.08 or abs(pitch - old_pitch) >= 0.08:
            return False
        roll_delta = abs((roll - old_roll + 180.0) % 360.0 - 180.0)
        if roll_delta >= 10.0:
            return False
    return True


def current_pose_signatures(states, active):
    if any(states[name].assessment is None for name in active):
        return None
    result = {
        name: states[name].assessment.pose_signature
        for name in active
    }
    return result if all(value is not None for value in result.values()) else None


def pose_geometry_changed(previous, current):
    if previous is None or current is None:
        return True
    _, _, area, yaw, pitch, roll = map(float, current)
    _, _, old_area, old_yaw, old_pitch, old_roll = map(float, previous)
    if min(area, old_area) <= 0 or max(area, old_area) / min(area, old_area) > 1.05:
        return True
    if abs(yaw - old_yaw) > 0.03 or abs(pitch - old_pitch) > 0.03:
        return True
    roll_delta = abs((roll - old_roll + 180.0) % 360.0 - 180.0)
    return roll_delta > 3.0


def assess_guided(image, detector, previous, now, max_motion_px):
    detections = filtered_detections(image, detector)
    annotated = image.copy()
    if detections:
        cv2.aruco.drawDetectedMarkers(
            annotated,
            [corners.reshape(1, 4, 2) for _, corners in detections],
            np.asarray([[marker_id] for marker_id, _ in detections], dtype=np.int32),
        )
    points = np.concatenate([corners for _, corners in detections]) if detections else None
    center = tuple(np.mean(points, axis=0)) if points is not None else None
    brightness, sharpness, clipped = image_metrics(image)
    base = ImageAssessment(
        tag_ids={marker_id for marker_id, _ in detections},
        brightness=brightness,
        sharpness=sharpness,
        clipped_percent=clipped,
        board_center=center,
        detected_at=now,
    )
    quad = projected_board(detections)
    if quad is not None:
        cv2.polylines(annotated, [quad.astype(np.int32)], True, (255, 0, 255), 2)
    area, position, scale, tilt = classify_geometry(
        quad, image.shape[1], image.shape[0]
    )
    signature = geometry_features(
        quad, image.shape[1], image.shape[0]
    )[-1]
    previous_base = previous.base if previous else None
    update_stability(previous_base, base, now, max_motion_px)
    previous_signature = previous.pose_signature if previous else None
    if (
        previous is not None
        and base.board_center is not None
        and pose_geometry_changed(previous_signature, signature)
    ):
        base.stable_since = now
    return GuidedAssessment(base, annotated, area, position, scale, tilt, signature)


def assessment_reasons(assessment, thresholds, args, now):
    if assessment is None:
        return ["detecting"]
    reasons = quality_reasons(assessment.base, thresholds)
    if now - assessment.base.detected_at > 2.0 / args.detection_fps + 0.1:
        reasons.append("stale frame")
    if assessment.area_ratio < args.min_area_ratio:
        reasons.append("move closer")
    elif assessment.area_ratio > args.max_area_ratio:
        reasons.append("move farther")
    if (
        assessment.base.stable_since is None
        or now - assessment.base.stable_since < args.settle_seconds
    ):
        reasons.append(f"hold still {args.settle_seconds:g}s")
    return reasons


def nearest_pair(buffers, max_skew_us):
    left_name, right_name = tuple(buffers)
    if not buffers[left_name] or not buffers[right_name]:
        return None, None
    closest_skew = None
    valid = []
    for left in buffers[left_name]:
        left_stamp = timestamp_us(left)
        for right in buffers[right_name]:
            right_stamp = timestamp_us(right)
            skew = abs(left_stamp - right_stamp)
            closest_skew = skew if closest_skew is None else min(closest_skew, skew)
            if skew <= max_skew_us:
                valid.append(
                    (min(left_stamp, right_stamp), -skew, left, right)
                )
    if not valid:
        return None, closest_skew
    # Save the newest valid pair, not an older pair captured before the board
    # became stable. The second key favors the lower skew at the same time.
    _, negative_skew, left, right = max(valid, key=lambda item: item[:2])
    return {left_name: left, right_name: right}, -negative_skew


def next_guidance(progress, mode):
    missing_positions = [
        index
        for index in POSITION_GUIDANCE_ORDER
        if index not in progress["positions"]
    ]
    if len(progress["positions"]) < 6 and missing_positions:
        return "NEXT: move board to " + POSITION_NAMES[missing_positions[0]]
    missing_scales = [name for name in SCALE_TARGETS if name not in progress["scales"]]
    if missing_scales:
        action = {"far": "farther", "mid": "middle distance", "near": "closer"}
        return "NEXT: move board " + action[missing_scales[0]]
    missing_tilts = [name for name in TILT_TARGETS if name not in progress["tilts"]]
    if missing_tilts:
        return "NEXT: add " + missing_tilts[0] + " view"
    target = 40 if mode == "intrinsics" else 20
    if progress["saved"] < target:
        return "NEXT: add a new non-duplicate pose"
    return "TARGET COMPLETE - collect extra varied views or quit"


def create_session(args, mode, active_cameras, device_id):
    started = datetime.now().astimezone()
    label = active_cameras[0] if mode == "intrinsics" else "-".join(active_cameras)
    session_dir = args.output_dir / f"{mode}_{label}_{started:%Y%m%d_%H%M%S_%f}"
    for name in active_cameras:
        (session_dir / name / "images").mkdir(parents=True)
        with (session_dir / name / "timestamps.csv").open("w", newline="", encoding="utf-8") as output:
            csv.writer(output).writerow(("frame_id", "timestamp_ns", "filename"))
    metadata = {
        "started_at": started.isoformat(timespec="microseconds"),
        "status": "running",
        "mode": mode,
        "active_cameras": list(active_cameras),
        "camera_mapping": {name: CAMERA_MAPPING[name] for name in active_cameras},
        "device_id": device_id,
        "resolution": list(SENSOR_RESOLUTION),
        "fps": args.fps,
        "sync_mode": args.sync_mode,
        "max_skew_us": args.max_skew_us if mode == "pair" else None,
        "target": {"family": "tag36h11", "cols": 6, "rows": 6, "tag_size_m": 0.055, "spacing": 0.3},
        "quality_thresholds": {
            "min_tags": args.min_tags,
            "min_sharpness": args.min_sharpness,
            "min_brightness": args.min_brightness,
            "max_brightness": args.max_brightness,
            "max_clipped_percent": args.max_clipped_percent,
        },
        "settle_seconds": args.settle_seconds,
        "manual_focus": args.manual_focus,
        "saved": 0,
        "positions": [],
        "scales": [],
        "tilts": [],
        "coverage": {
            name: {"positions": [], "scales": [], "tilts": []}
            for name in active_cameras
        },
        "signatures": [],
    }
    write_json_atomic(session_dir / "session.json", metadata)
    return session_dir, metadata


def append_timestamp(path, frame_id, frame):
    filename = f"{frame_id:06d}.png"
    with path.open("a", newline="", encoding="utf-8") as output:
        csv.writer(output).writerow((frame_id, timestamp_us(frame) * 1000, filename))


def save_frames(session_dir, frames, states, metadata):
    frame_id = int(metadata["saved"])
    filename = f"{frame_id:06d}.png"
    staged = []
    committed = []
    try:
        for name, frame in frames.items():
            state = states[name]
            if state.frame is None or timestamp_us(state.frame) != timestamp_us(frame):
                raise RuntimeError(f"{name}保存帧与质量评估帧不一致")
            image = frame.getCvFrame()
            if (image.shape[1], image.shape[0]) != SENSOR_RESOLUTION:
                raise RuntimeError(f"{name}不是1280x800全分辨率图像")
            output_path = session_dir / name / "images" / filename
            temporary = output_path.with_name(f".{filename}.tmp.png")
            if not cv2.imwrite(str(temporary), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise RuntimeError(f"写入失败：{temporary}")
            staged.append((temporary, output_path))
        for temporary, output_path in staged:
            temporary.replace(output_path)
            committed.append(output_path)
    except Exception:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        for output_path in committed:
            output_path.unlink(missing_ok=True)
        raise
    for name, frame in frames.items():
        append_timestamp(session_dir / name / "timestamps.csv", frame_id, frame)

    coverage = metadata.setdefault("coverage", {})
    for name in frames:
        assessment = states[name].assessment
        camera_coverage = coverage.setdefault(
            name, {"positions": [], "scales": [], "tilts": []}
        )
        for key, value in (
            ("positions", assessment.position_bin),
            ("scales", assessment.scale_class),
            ("tilts", assessment.tilt_class),
        ):
            if value is not None and value not in camera_coverage[key]:
                camera_coverage[key].append(value)

    primary = states[next(iter(frames))].assessment
    metadata["saved"] = frame_id + 1
    for key, value in (
        ("positions", primary.position_bin),
        ("scales", primary.scale_class),
        ("tilts", primary.tilt_class),
    ):
        if value is not None and value not in metadata[key]:
            metadata[key].append(value)
    metadata["last_timestamps_us"] = {name: timestamp_us(frame) for name, frame in frames.items()}
    write_json_atomic(session_dir / "session.json", metadata)


def assess_selected_frames(states, frames, detector, now, max_motion_px):
    """Assess the exact full-resolution frames that SPACE will save."""
    for name, frame in frames.items():
        full = frame.getCvFrame()
        preview = cv2.resize(full, (640, 400), interpolation=cv2.INTER_AREA)
        states[name].frame = frame
        states[name].preview = preview
        states[name].assessment = assess_guided(
            preview,
            detector,
            states[name].assessment,
            now,
            max_motion_px,
        )
        states[name].last_detection_at = now


def frame_group_key(frames):
    if not frames:
        return None
    return tuple((name, timestamp_us(frame)) for name, frame in frames.items())


def fit_view(image, width, height):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if image is None:
        cv2.putText(canvas, "NO FRAME", (width // 2 - 80, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 190, 255), 2)
        return canvas
    scale = min(width / image.shape[1], height / image.shape[0])
    size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    x = (width - size[0]) // 2
    y = (height - size[1]) // 2
    canvas[y:y + size[1], x:x + size[0]] = resized
    return canvas


def draw_coverage(canvas, positions, origin):
    x0, y0 = origin
    positions = set(positions)
    for index in range(9):
        row, column = divmod(index, 3)
        left = x0 + column * 42
        top = y0 + row * 42
        color = GRID_COLORS[1] if index in positions else GRID_COLORS[0]
        cv2.rectangle(canvas, (left, top), (left + 36, top + 36), color, cv2.FILLED)
        cv2.putText(canvas, str(index + 1), (left + 11, top + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def build_view(
    states,
    active,
    thresholds,
    args,
    metadata,
    now,
    skew_us,
    frames_available=True,
):
    panel_width = DISPLAY_SIZE[0] // len(active)
    panels = []
    all_reasons = {}
    for name in active:
        state = states[name]
        assessment = state.assessment
        source = assessment.annotated if assessment else state.preview
        panel = fit_view(source, panel_width, DISPLAY_SIZE[1])
        reasons = assessment_reasons(assessment, thresholds, args, now)
        all_reasons[name] = reasons
        base = assessment.base if assessment else None
        lines = (
            f"{name}/{CAMERA_MAPPING[name]} {state.fps:4.1f} FPS",
            f"tags={len(base.tag_ids) if base else 0:02d} sharp={base.sharpness if base else 0:6.1f}",
            f"area={assessment.area_ratio if assessment else 0:.3f} pos={POSITION_NAMES[assessment.position_bin] if assessment and assessment.position_bin is not None else '?'}",
            f"scale={assessment.scale_class if assessment else '?'} tilt={assessment.tilt_class if assessment else '?'}",
            "GOOD" if not reasons else ", ".join(reasons),
        )
        for index, line in enumerate(lines):
            cv2.putText(panel, line, (12, 28 + index * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 220, 60) if not reasons else (0, 190, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    body = cv2.hconcat(panels)
    footer = np.full((170, body.shape[1], 3), 25, dtype=np.uint8)
    signatures = current_pose_signatures(states, active)
    duplicate = bool(
        signatures
        and any(
            pose_signatures_duplicate(signatures, saved)
            for saved in metadata.get("signatures", [])
            if isinstance(saved, dict)
        )
    )
    ready = frames_available and not any(all_reasons.values()) and not duplicate
    if len(active) == 2:
        ready = ready and skew_us is not None and skew_us <= args.max_skew_us
    reason = "READY - PRESS SPACE" if ready else "BLOCKED"
    if not frames_available:
        reason += ": waiting for assessed frame"
    elif duplicate:
        reason += ": duplicate pose"
    elif len(active) == 2 and (skew_us is None or skew_us > args.max_skew_us):
        reason += f": pair skew {skew_us if skew_us is not None else '?'}>{args.max_skew_us} us"
    elif any(all_reasons.values()):
        reason += ": " + " | ".join(f"{name}:{','.join(values)}" for name, values in all_reasons.items() if values)
    if len(active) == 2:
        skew_text = "waiting" if skew_us is None else f"{skew_us:.0f} us"
        reason += f"  [{args.sync_mode.upper()} pair skew={skew_text}; limit={args.max_skew_us} us]"
    coverage = metadata.get("coverage", {})
    guidance_camera = min(
        active,
        key=lambda name: (
            len(coverage.get(name, {}).get("positions", [])),
            len(coverage.get(name, {}).get("scales", [])),
            len(coverage.get(name, {}).get("tilts", [])),
        ),
    )
    camera_progress = coverage.get(guidance_camera, {})
    progress = {
        "saved": metadata["saved"],
        "positions": set(camera_progress.get("positions", metadata["positions"])),
        "scales": set(camera_progress.get("scales", metadata["scales"])),
        "tilts": set(camera_progress.get("tilts", metadata["tilts"])),
    }
    guidance = next_guidance(progress, metadata["mode"])
    if len(active) == 2:
        guidance = f"{guidance_camera}: {guidance}"
    cv2.putText(footer, reason, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 220, 60) if ready else (0, 190, 255), 2, cv2.LINE_AA)
    cv2.putText(footer, guidance, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 220, 80), 2, cv2.LINE_AA)
    coverage_text = " ".join(
        f"{name}:pos{len(coverage.get(name, {}).get('positions', []))}/9"
        for name in active
    )
    cv2.putText(footer, f"saved={metadata['saved']} {coverage_text} scales={metadata['scales']}  SPACE=save Q/ESC=quit", (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
    primary_positions = coverage.get(active[0], {}).get(
        "positions", metadata["positions"]
    )
    draw_coverage(footer, primary_positions, (body.shape[1] - 140, 18))
    return cv2.vconcat((body, footer)), ready


def validate_socket_mapping(sockets, active):
    by_name = {socket.name: socket for socket in sockets}
    missing = [name for name in active if CAMERA_MAPPING[name] not in by_name]
    if missing:
        raise RuntimeError(f"缺少相机：{missing}")
    return {name: by_name[CAMERA_MAPPING[name]] for name in active}


def run(args, mode, active):
    thresholds = thresholds_from_args(args)
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        raise RuntimeError("未发现OAK设备")
    with dai.Device(devices[0]) as device, dai.Pipeline(device) as pipeline:
        sockets = validate_socket_mapping(device.getConnectedCameras(), active)
        queues = {}
        for name in active:
            camera = pipeline.create(dai.node.Camera)
            camera.setSensorType(dai.CameraSensorType.COLOR)
            camera.build(sockets[name], sensorResolution=SENSOR_RESOLUTION, sensorFps=args.fps)
            if args.sync_mode == "fsin":
                camera.initialControl.setFrameSyncMode(
                    dai.CameraControl.FrameSyncMode.INPUT
                )
            if args.manual_focus is not None:
                camera.initialControl.setManualFocus(args.manual_focus)
            output = camera.requestFullResolutionOutput(type=dai.ImgFrame.Type.NV12, fps=args.fps)
            queue = output.createOutputQueue()
            queue.setMaxSize(12 if mode == "pair" else 2)
            queue.setBlocking(False)
            queues[name] = queue
        pipeline.start()
        session_dir, metadata = create_session(args, mode, active, device.getDeviceId())
        states = {name: StreamState() for name in active}
        buffers = {name: deque(maxlen=12) for name in active}
        detector = create_detector()
        detection_period = 1.0 / args.detection_fps
        current_frames = None
        current_skew = None
        last_group_detection_at = float("-inf")
        last_assessed_group_key = None
        last_sync_report_at = float("-inf")
        print(f"数据目录：{session_dir}")
        print(f"模式={mode}，相机={active}，只传输1280x800原始流")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        status = "completed"
        error = None
        try:
            while pipeline.isRunning():
                now = time.monotonic()
                for name, queue in queues.items():
                    messages = queue.tryGetAll()
                    for frame in messages:
                        buffers[name].append(frame)
                        states[name].update_rate(timestamp_us(frame))
                if mode == "intrinsics":
                    candidate_frames = (
                        {active[0]: buffers[active[0]][-1]}
                        if buffers[active[0]]
                        else None
                    )
                    candidate_skew = 0
                else:
                    candidate_frames, candidate_skew = nearest_pair(
                        buffers, args.max_skew_us
                    )
                if mode == "pair" and now - last_sync_report_at >= 1.0:
                    outcome = "PASS" if candidate_frames is not None else "FAIL"
                    skew_text = (
                        "waiting for both streams"
                        if candidate_skew is None
                        else f"{candidate_skew:.0f} us"
                    )
                    print(
                        f"同步检查 {args.sync_mode.upper()} {active[0]}-{active[1]}: "
                        f"{outcome}; pair skew={skew_text}; "
                        f"limit={args.max_skew_us} us",
                        flush=True,
                    )
                    last_sync_report_at = now
                if (
                    candidate_frames
                    and frame_group_key(candidate_frames) != last_assessed_group_key
                    and now - last_group_detection_at >= detection_period
                ):
                    assess_selected_frames(
                        states,
                        candidate_frames,
                        detector,
                        now,
                        args.max_motion_px,
                    )
                    current_frames = candidate_frames
                    current_skew = candidate_skew
                    last_group_detection_at = now
                    last_assessed_group_key = frame_group_key(candidate_frames)
                view, ready = build_view(
                    states,
                    active,
                    thresholds,
                    args,
                    metadata,
                    now,
                    current_skew,
                    frames_available=current_frames is not None,
                )
                cv2.imshow(WINDOW_NAME, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    if not ready or not current_frames or any(frame is None for frame in current_frames.values()):
                        print("保存被拒绝：请按界面提示调整")
                        continue
                    previous = metadata.get("last_timestamps_us", {})
                    if any(timestamp_us(frame) <= previous.get(name, -1) for name, frame in current_frames.items()):
                        print("保存被拒绝：当前帧已经保存过")
                        continue
                    signatures = current_pose_signatures(states, active)
                    save_frames(session_dir, current_frames, states, metadata)
                    if signatures is not None:
                        metadata["signatures"].append(
                            {
                                name: list(signature)
                                for name, signature in signatures.items()
                            }
                        )
                        write_json_atomic(session_dir / "session.json", metadata)
                    for buffer in buffers.values():
                        buffer.clear()
                    current_frames = None
                    current_skew = None
                    print(f"已保存{metadata['saved']:03d}，{next_guidance({'saved': metadata['saved'], 'positions': set(metadata['positions']), 'scales': set(metadata['scales']), 'tilts': set(metadata['tilts'])}, mode)}")
        except KeyboardInterrupt:
            status = "interrupted"
            print("\n采集已停止")
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
            metadata["status"] = status
            metadata["ended_at"] = datetime.now().astimezone().isoformat(timespec="microseconds")
            metadata["error"] = error
            write_json_atomic(session_dir / "session.json", metadata)
    return session_dir


def cli_main(args, mode, active):
    try:
        run(args, mode, active)
    except (RuntimeError, OSError, cv2.error) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0
