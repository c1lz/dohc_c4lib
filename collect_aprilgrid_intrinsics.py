#!/usr/bin/env python3
"""Collect guided AprilGrid views for one camera's intrinsics."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dohc2_c4lib.aprilgrid_common import CAMERA_IDS
from dohc2_c4lib.guided_capture import add_common_arguments, cli_main, validate_args


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="逐路采集AprilGrid内参数据")
    parser.add_argument("--camera", choices=CAMERA_IDS, required=True)
    add_common_arguments(parser)
    return validate_args(parser, parser.parse_args(argv))


def main(argv=None):
    args = parse_args(argv)
    return cli_main(args, "intrinsics", (args.camera,))


if __name__ == "__main__":
    raise SystemExit(main())
