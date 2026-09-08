#!/usr/bin/env python3
"""Generate a Kalibr imu.yaml from a stationary BNO086 Allan capture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dohc2_c4lib.imu_allan import estimate_imu, write_kalibr_imu_yaml


def main(argv=None):
    parser = argparse.ArgumentParser(description="从 BNO086 Allan 数据生成 Kalibr imu.yaml")
    parser.add_argument("session_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "imu_bno086.yaml",
    )
    parser.add_argument("--topic", default="/mavros/imu/data_raw")
    args = parser.parse_args(argv)
    estimate = estimate_imu(args.session_dir / "accel.csv", args.session_dir / "gyro.csv")
    write_kalibr_imu_yaml(args.output, estimate, args.topic)
    report = {
        "duration_s": estimate.duration_s,
        "sample_rate_hz": estimate.sample_rate_hz,
        "accelerometer_noise_density": estimate.accel_noise_density,
        "accelerometer_random_walk": estimate.accel_random_walk,
        "gyroscope_noise_density": estimate.gyro_noise_density,
        "gyroscope_random_walk": estimate.gyro_random_walk,
        "axis_std": estimate.axes_std,
        "imu_yaml": str(args.output),
    }
    report_path = args.output.with_suffix(".allan_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
