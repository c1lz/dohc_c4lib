#!/usr/bin/env python3
"""Collect four color camera streams and RAW IMU in Extended EuRoC format."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dohc2_c4lib.euroc_capture import CAMERAS, main

if __name__ == "__main__":
    raise SystemExit(main(CAMERAS))
