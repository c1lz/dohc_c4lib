"""BNO086 raw-IMU Allan deviation estimation utilities."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AllanEstimate:
    sample_rate_hz: float
    accel_noise_density: float
    accel_random_walk: float
    gyro_noise_density: float
    gyro_random_walk: float
    duration_s: float
    axes_std: dict[str, list[float]]


def load_imu_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps, values = [], []
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            timestamps.append(int(row["timestamp_ns"]))
            values.append((float(row["x"]), float(row["y"]), float(row["z"])))
    if len(timestamps) < 100:
        raise ValueError(f"{path} 的 IMU 样本不足")
    return np.asarray(timestamps, dtype=np.int64), np.asarray(values, dtype=np.float64)


def validate_timestamps(timestamps_ns: np.ndarray) -> float:
    deltas_s = np.diff(timestamps_ns).astype(np.float64) / 1e9
    if np.any(deltas_s <= 0):
        raise ValueError("IMU 时间戳必须严格递增")
    # BNO086 reports are delivered in batches and their individual device
    # timestamp intervals have visible scheduling jitter.  Use the complete
    # device-clock span rather than the median interval: the latter can bias
    # the inferred rate significantly while the long-run count/span rate is
    # what Kalibr's `update_rate` represents.
    period_s = float((timestamps_ns[-1] - timestamps_ns[0]) / (len(timestamps_ns) - 1)) / 1e9
    if period_s <= 0:
        raise ValueError("无效 IMU 采样周期")
    return 1.0 / period_s


def resample_uniform(
    timestamps_ns: np.ndarray, values: np.ndarray, sample_rate_hz: float
) -> np.ndarray:
    """Linearly resample device-timestamped samples onto the median-rate grid."""
    start_s = timestamps_ns[0] / 1e9
    end_s = timestamps_ns[-1] / 1e9
    count = int(math.floor((end_s - start_s) * sample_rate_hz)) + 1
    grid_s = start_s + np.arange(count, dtype=np.float64) / sample_rate_hz
    source_s = timestamps_ns.astype(np.float64) / 1e9
    return np.column_stack(
        [np.interp(grid_s, source_s, values[:, axis]) for axis in range(3)]
    )


def overlapping_allan_deviation(
    samples: np.ndarray, sample_rate_hz: float, taus_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid tau values and overlapping Allan deviations for three axes."""
    count = len(samples)
    cumulative = np.vstack((np.zeros(3), np.cumsum(samples, axis=0)))
    valid_taus, deviations = [], []
    used_clusters = set()
    for tau_s in taus_s:
        cluster = max(1, int(round(tau_s * sample_rate_hz)))
        if cluster in used_clusters or 2 * cluster >= count:
            continue
        used_clusters.add(cluster)
        averages = (cumulative[cluster:] - cumulative[:-cluster]) / cluster
        difference = averages[cluster:] - averages[:-cluster]
        deviations.append(np.sqrt(0.5 * np.mean(difference * difference, axis=0)))
        valid_taus.append(cluster / sample_rate_hz)
    if not deviations:
        raise ValueError("序列太短，无法计算 Allan 方差")
    return np.asarray(valid_taus), np.asarray(deviations)


def _coefficient_for_slope(
    taus_s: np.ndarray,
    deviations: np.ndarray,
    target_slope: float,
    transform,
    tau_min_s: float,
    tau_max_s: float,
) -> float:
    usable = (taus_s >= tau_min_s) & (taus_s <= tau_max_s)
    if not np.any(usable):
        raise ValueError("Allan 曲线没有可用于拟合的时间尺度")
    coefficients = []
    log_tau = np.log(taus_s)
    for axis in range(deviations.shape[1]):
        log_deviation = np.log(np.maximum(deviations[:, axis], np.finfo(float).tiny))
        slopes = np.gradient(log_deviation, log_tau)
        selected = usable & (np.abs(slopes - target_slope) <= 0.30)
        if not np.any(selected):
            selected = usable
        coefficients.append(
            float(np.median(transform(deviations[selected, axis], taus_s[selected])))
        )
    # Kalibr accepts a scalar; using the worst axis avoids overconfident weighting.
    return float(max(coefficients))


def estimate_sensor(
    timestamps_ns: np.ndarray, values: np.ndarray
) -> tuple[float, float, float, list[float]]:
    sample_rate_hz = validate_timestamps(timestamps_ns)
    uniform_values = resample_uniform(timestamps_ns, values, sample_rate_hz)
    duration_s = len(uniform_values) / sample_rate_hz
    if duration_s < 120.0:
        raise ValueError("Allan 方差至少需要 120 秒静止原始 IMU 数据")
    tau_max_s = min(duration_s / 10.0, 180.0)
    taus_s = np.unique(
        np.geomspace(1.0 / sample_rate_hz, tau_max_s, num=80).round(8)
    )
    taus_s, deviations = overlapping_allan_deviation(uniform_values, sample_rate_hz, taus_s)
    noise_density = _coefficient_for_slope(
        taus_s,
        deviations,
        target_slope=-0.5,
        transform=lambda sigma, tau: sigma * np.sqrt(tau),
        tau_min_s=1.0 / sample_rate_hz,
        tau_max_s=min(0.5, duration_s / 20.0),
    )
    random_walk = _coefficient_for_slope(
        taus_s,
        deviations,
        target_slope=0.5,
        transform=lambda sigma, tau: sigma * np.sqrt(3.0 / tau),
        tau_min_s=max(10.0, duration_s / 100.0),
        tau_max_s=tau_max_s,
    )
    return sample_rate_hz, noise_density, random_walk, uniform_values.std(axis=0).tolist()


def estimate_imu(accel_csv: Path, gyro_csv: Path) -> AllanEstimate:
    accel_timestamps, accel_values = load_imu_csv(accel_csv)
    gyro_timestamps, gyro_values = load_imu_csv(gyro_csv)
    accel_rate, accel_noise, accel_walk, accel_std = estimate_sensor(
        accel_timestamps, accel_values
    )
    gyro_rate, gyro_noise, gyro_walk, gyro_std = estimate_sensor(
        gyro_timestamps, gyro_values
    )
    duration_s = min(
        (accel_timestamps[-1] - accel_timestamps[0]) / 1e9,
        (gyro_timestamps[-1] - gyro_timestamps[0]) / 1e9,
    )
    return AllanEstimate(
        sample_rate_hz=(accel_rate + gyro_rate) / 2.0,
        accel_noise_density=accel_noise,
        accel_random_walk=accel_walk,
        gyro_noise_density=gyro_noise,
        gyro_random_walk=gyro_walk,
        duration_s=duration_s,
        axes_std={"accel": accel_std, "gyro": gyro_std},
    )


def write_kalibr_imu_yaml(path: Path, estimate: AllanEstimate, topic: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Generated from stationary OAK-4P BNO086 RAW IMU data using overlapping Allan deviation.\n"
        f"# Duration: {estimate.duration_s:.1f} s; observed rate: {estimate.sample_rate_hz:.6f} Hz\n"
        f"accelerometer_noise_density: {estimate.accel_noise_density:.12g}\n"
        f"accelerometer_random_walk: {estimate.accel_random_walk:.12g}\n"
        f"gyroscope_noise_density: {estimate.gyro_noise_density:.12g}\n"
        f"gyroscope_random_walk: {estimate.gyro_random_walk:.12g}\n"
        f"rostopic: '{topic}'\n"
        f"update_rate: {round(estimate.sample_rate_hz):.1f}\n",
        encoding="utf-8",
    )
