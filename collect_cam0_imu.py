#!/usr/bin/env python3
"""Collect CAM_A color images and RAW IMU, or import independent Mocap CSV."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dohc2_c4lib.euroc_capture import main

if __name__ == "__main__":
    raise SystemExit(main(("cam0",)))
