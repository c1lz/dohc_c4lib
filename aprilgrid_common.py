#!/usr/bin/env python3
"""Shared AprilGrid detection and dataset helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np


CAMERA_MAPPING = {
    "cam0": "CAM_A",
    "cam1": "CAM_B",
    "cam2": "CAM_C",
    "cam3": "CAM_D",
}
CAMERA_IDS = tuple(CAMERA_MAPPING)
ADJACENT_PAIRS = (
    ("cam0", "cam1"),
    ("cam1", "cam2"),
    ("cam2", "cam3"),
    ("cam3", "cam0"),
)
APRILGRID_IDS = frozenset(range(36))
SENSOR_RESOLUTION = (1280, 800)
PREVIEW_RESOLUTION = (640, 400)


@dataclass(frozen=True)
class QualityThresholds:
    min_tags: int = 6
    min_sharpness: float = 80.0
    min_brightness: float = 45.0
    max_brightness: float = 210.0
    max_clipped_percent: float = 5.0


@dataclass
class ImageAssessment:
    tag_ids: set[int]
    brightness: float
    sharpness: float
    clipped_percent: float
    board_center: tuple[float, float] | None
    detected_at: float = 0.0
    stable_since: float | None = None

    def serializable(self) -> dict[str, object]:
        result = asdict(self)
        result["tag_ids"] = sorted(self.tag_ids)
        return result


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reason: str
    skew_us: int | None
    qualified_cameras: tuple[str, ...]
    stable_cameras: tuple[str, ...]


def create_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36h11
    )
    parameters = cv2.aruco.DetectorParameters()
    # Kalibr's Tag36h11 target uses a two-bit black marker border. OpenCV's
    # default is one bit and only detected 0-2 tags on the physical board.
    parameters.markerBorderBits = 2
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeStep = 1
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def timestamp_us(frame) -> int:
    timestamp = frame.getTimestampDevice()
    return (
        (timestamp.days * 86_400 + timestamp.seconds) * 1_000_000
        + timestamp.microseconds
    )


def image_metrics(image: np.ndarray) -> tuple[float, float, float]:
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image
    )
    valid = gray > 8
    if not np.any(valid):
        return 0.0, 0.0, 100.0
    pixels = gray[valid]
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    clipped = np.count_nonzero((pixels <= 12) | (pixels >= 250))
    return (
        float(np.mean(pixels)),
        float(np.var(laplacian[valid])),
        100.0 * float(clipped) / float(pixels.size),
    )


def detect_aprilgrid(
    image: np.ndarray,
    detector,
) -> tuple[np.ndarray, set[int], tuple[float, float] | None]:
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image
    )
    corners, ids, _ = detector.detectMarkers(gray)
    valid_ids: set[int] = set()
    valid_corners = []
    valid_id_rows = []
    centers = []
    if ids is not None:
        for corner, marker_id in zip(corners, ids.reshape(-1)):
            marker_id = int(marker_id)
            if marker_id not in APRILGRID_IDS or marker_id in valid_ids:
                continue
            valid_ids.add(marker_id)
            valid_corners.append(corner)
            valid_id_rows.append([marker_id])
            centers.append(np.mean(np.asarray(corner).reshape(-1, 2), axis=0))
    annotated = (
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 2
        else image.copy()
    )
    if valid_corners:
        cv2.aruco.drawDetectedMarkers(
            annotated,
            valid_corners,
            np.asarray(valid_id_rows, dtype=np.int32),
        )
    center = None
    if centers:
        mean_center = np.mean(np.asarray(centers), axis=0)
        center = (float(mean_center[0]), float(mean_center[1]))
    return annotated, valid_ids, center


def assess_image(image: np.ndarray, detector, now: float = 0.0):
    annotated, tag_ids, center = detect_aprilgrid(image, detector)
    brightness, sharpness, clipped = image_metrics(image)
    return annotated, ImageAssessment(
        tag_ids=tag_ids,
        brightness=brightness,
        sharpness=sharpness,
        clipped_percent=clipped,
        board_center=center,
        detected_at=now,
    )


def quality_reasons(
    assessment: ImageAssessment,
    thresholds: QualityThresholds,
) -> list[str]:
    reasons = []
    if len(assessment.tag_ids) < thresholds.min_tags:
        reasons.append(f"tags<{thresholds.min_tags}")
    if assessment.sharpness < thresholds.min_sharpness:
        reasons.append("blur")
    if not (
        thresholds.min_brightness
        <= assessment.brightness
        <= thresholds.max_brightness
    ):
        reasons.append("exposure")
    if assessment.clipped_percent > thresholds.max_clipped_percent:
        reasons.append("clipped")
    return reasons


def update_stability(
    previous: ImageAssessment | None,
    current: ImageAssessment,
    now: float,
    max_motion_px: float,
) -> None:
    current.detected_at = now
    if current.board_center is None:
        current.stable_since = None
        return
    if previous is None or previous.board_center is None:
        current.stable_since = now
        return
    movement = float(
        np.linalg.norm(
            np.asarray(current.board_center) - np.asarray(previous.board_center)
        )
    )
    current.stable_since = (
        previous.stable_since
        if movement <= max_motion_px and previous.stable_since is not None
        else now
    )


def evaluate_group(
    timestamps_us: Mapping[str, int | None],
    assessments: Mapping[str, ImageAssessment | None],
    now: float,
    thresholds: QualityThresholds,
    max_skew_us: int,
    settle_seconds: float,
    max_assessment_age: float,
) -> GateDecision:
    missing = [name for name in CAMERA_IDS if timestamps_us.get(name) is None]
    if missing:
        return GateDecision(
            False,
            "missing full frame: " + ",".join(missing),
            None,
            (),
            (),
        )
    values = [int(timestamps_us[name]) for name in CAMERA_IDS]
    skew_us = max(values) - min(values)
    if skew_us > max_skew_us:
        return GateDecision(
            False,
            f"timestamp skew {skew_us}>{max_skew_us} us",
            skew_us,
            (),
            (),
        )

    qualified = []
    stable = []
    for name in CAMERA_IDS:
        assessment = assessments.get(name)
        if assessment is None or now - assessment.detected_at > max_assessment_age:
            continue
        if quality_reasons(assessment, thresholds):
            continue
        qualified.append(name)
        if (
            assessment.stable_since is not None
            and now - assessment.stable_since >= settle_seconds
        ):
            stable.append(name)
    if not qualified:
        return GateDecision(
            False,
            "no camera passes AprilGrid quality checks",
            skew_us,
            (),
            (),
        )
    if not stable:
        return GateDecision(
            False,
            f"hold board still for {settle_seconds:g} s",
            skew_us,
            tuple(qualified),
            (),
        )
    return GateDecision(
        True,
        "ready",
        skew_us,
        tuple(qualified),
        tuple(stable),
    )


def visible_pairs(camera_names) -> tuple[str, ...]:
    visible = set(camera_names)
    return tuple(
        f"{left}-{right}"
        for left, right in ADJACENT_PAIRS
        if left in visible and right in visible
    )


def write_json_atomic(path: Path, data: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
