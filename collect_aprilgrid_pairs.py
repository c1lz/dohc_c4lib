#!/usr/bin/env python3
"""Collect guided AprilGrid covisible views for any two different cameras."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dohc2_c4lib.aprilgrid_common import CAMERA_IDS
from dohc2_c4lib.guided_capture import add_common_arguments, cli_main, validate_args


# Keep the order supplied by the operator in session metadata.  This also keeps
# legacy labels such as cam3-cam0 valid while allowing a changed rig topology.
PAIRS = tuple(
    f"{left}-{right}"
    for left in CAMERA_IDS
    for right in CAMERA_IDS
    if left != right
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="逐对采集AprilGrid共视外参数据（任意两路不同相机）")
    parser.add_argument("--pair", choices=PAIRS, required=True)
    add_common_arguments(parser)
    return validate_args(parser, parser.parse_args(argv))


def main(argv=None):
    args = parse_args(argv)
    return cli_main(args, "pair", tuple(args.pair.split("-")))


if __name__ == "__main__":
    raise SystemExit(main())
