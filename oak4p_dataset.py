#!/usr/bin/env python3
"""Shared data model and synchronization helpers for OAK-4P recordings."""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dohc2_c4lib.aprilgrid_common import CAMERA_IDS, write_json_atomic


CAMERA_CSV_FIELDS = (
    "timestamp_ns",
    "filename",
    "exposure_us",
    "iso",
    "sequence_num",
)
IMU_CSV_FIELDS = ("timestamp_ns", "x", "y", "z", "sequence_num")
GROUP_CSV_FIELDS = (
    "group_id",
    "reference_timestamp_ns",
    "skew_us",
    *(f"{name}_timestamp_ns" for name in CAMERA_IDS),
)
STRICT_SYNC_US = 1_000
DEFAULT_MAX_PIXEL_PHASE_SPREAD = 5.0


@dataclass(frozen=True)
class CameraRow:
    timestamp_ns: int
    filename: str
    exposure_us: int
    iso: int
    sequence_num: int


@dataclass(frozen=True)
class ImuRow:
    timestamp_ns: int
    x: float
    y: float
    z: float
    sequence_num: int


def pixel_phase_metrics(image: np.ndarray) -> dict[str, object]:
    """Measure residual 2x2 pixel-phase structure in a grayscale image.

    Undemosaiced Bayer data has a large, sensor-aligned difference between the
    four (row parity, column parity) populations.  A wide central crop makes
    the statistic insensitive to ordinary scene structure while retaining the
    fixed 2x2 pattern.
    """
    if image.ndim != 2:
        raise ValueError("pixel phase analysis requires a single-channel image")
    height, width = image.shape
    if height < 8 or width < 8:
        raise ValueError("pixel phase analysis requires an image of at least 8x8")
    margin_y = height // 10
    margin_x = width // 10
    crop = image[margin_y : height - margin_y, margin_x : width - margin_x]
    phase_means = [
        float(np.mean(crop[row_phase::2, column_phase::2], dtype=np.float64))
        for row_phase in range(2)
        for column_phase in range(2)
    ]
    return {
        "phase_means": phase_means,
        "spread": max(phase_means) - min(phase_means),
    }


def timedelta_ns(value) -> int:
    """Convert DepthAI's timedelta-like device timestamp to integer ns."""
    return (
        (int(value.days) * 86_400 + int(value.seconds)) * 1_000_000_000
        + int(value.microseconds) * 1_000
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def load_camera_rows(session_dir: Path, camera_id: str) -> list[CameraRow]:
    path = session_dir / camera_id / "data.csv"
    rows = read_csv_rows(path)
    return [
        CameraRow(
            timestamp_ns=int(row["timestamp_ns"]),
            filename=row["filename"],
            exposure_us=int(row["exposure_us"]),
            iso=int(row["iso"]),
            sequence_num=int(row["sequence_num"]),
        )
        for row in rows
    ]


def load_imu_rows(path: Path) -> list[ImuRow]:
    return [
        ImuRow(
            timestamp_ns=int(row["timestamp_ns"]),
            x=float(row["x"]),
            y=float(row["y"]),
            z=float(row["z"]),
            sequence_num=int(row["sequence_num"]),
        )
        for row in read_csv_rows(path)
    ]


def nearest_index(rows: Sequence[CameraRow], target_ns: int, start: int) -> int:
    if not rows:
        return -1
    index = min(max(start, 0), len(rows) - 1)
    while (
        index + 1 < len(rows)
        and abs(rows[index + 1].timestamp_ns - target_ns)
        <= abs(rows[index].timestamp_ns - target_ns)
    ):
        index += 1
    return index


def build_groups(
    camera_rows: Mapping[str, Sequence[CameraRow]], max_skew_us: int
) -> list[dict[str, int]]:
    """Match every cam0 frame to unused, nearest frames from all other cameras."""
    if max_skew_us <= 0:
        raise ValueError("max_skew_us must be positive")
    if any(not camera_rows.get(name) for name in CAMERA_IDS):
        return []
    cursors = {name: 0 for name in CAMERA_IDS[1:]}
    groups: list[dict[str, int]] = []
    limit_ns = max_skew_us * 1_000
    for reference in camera_rows[CAMERA_IDS[0]]:
        selected = {CAMERA_IDS[0]: reference}
        candidate_indices: dict[str, int] = {}
        for name in CAMERA_IDS[1:]:
            rows = camera_rows[name]
            index = nearest_index(rows, reference.timestamp_ns, cursors[name])
            if index < cursors[name] or index < 0:
                break
            selected[name] = rows[index]
            candidate_indices[name] = index
        if len(selected) != len(CAMERA_IDS):
            continue
        stamps = [row.timestamp_ns for row in selected.values()]
        if max(stamps) - min(stamps) > limit_ns:
            continue
        group = {
            "group_id": len(groups),
            "reference_timestamp_ns": reference.timestamp_ns,
            "skew_us": math.ceil((max(stamps) - min(stamps)) / 1_000),
        }
        group.update(
            {f"{name}_timestamp_ns": selected[name].timestamp_ns for name in CAMERA_IDS}
        )
        groups.append(group)
        for name, index in candidate_indices.items():
            cursors[name] = index + 1
    return groups


def write_groups(path: Path, groups: Iterable[Mapping[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=GROUP_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(groups)
    temporary.replace(path)


def percentile(values: Sequence[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def interpolate_accel(
    accel: Sequence[ImuRow], gyro_timestamps_ns: Iterable[int]
) -> tuple[list[tuple[int, float, float, float]], int]:
    """Linearly interpolate acceleration at gyro timestamps; never extrapolate."""
    if len(accel) < 2:
        return [], len(list(gyro_timestamps_ns))
    output: list[tuple[int, float, float, float]] = []
    dropped = 0
    index = 0
    for timestamp_ns in gyro_timestamps_ns:
        while index + 1 < len(accel) and accel[index + 1].timestamp_ns < timestamp_ns:
            index += 1
        if (
            timestamp_ns < accel[0].timestamp_ns
            or timestamp_ns > accel[-1].timestamp_ns
            or index + 1 >= len(accel)
        ):
            dropped += 1
            continue
        left, right = accel[index], accel[index + 1]
        if timestamp_ns == left.timestamp_ns:
            ratio = 0.0
        elif right.timestamp_ns == left.timestamp_ns:
            dropped += 1
            continue
        else:
            ratio = (timestamp_ns - left.timestamp_ns) / (
                right.timestamp_ns - left.timestamp_ns
            )
        output.append(
            (
                timestamp_ns,
                left.x + ratio * (right.x - left.x),
                left.y + ratio * (right.y - left.y),
                left.z + ratio * (right.z - left.z),
            )
        )
    return output, dropped


def finalize_session_metadata(session_dir: Path, updates: Mapping[str, object]) -> dict:
    path = session_dir / "session.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata.update(updates)
    write_json_atomic(path, metadata)
    return metadata
