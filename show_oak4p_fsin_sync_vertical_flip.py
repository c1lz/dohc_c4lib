#!/usr/bin/env python3
"""按安装方向翻转四路相机画面的 OAK-4P FSIN 同步验证器。

CAM_A、CAM_B、CAM_C、CAM_D 四路图像均同时上下和左右翻转；相机映射、
设备时间戳、最近邻配组、FPS、丢帧计数与同步 PASS/FAIL 逻辑均复用
``show_oak4p_fsin_sync.py``。
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import cv2

# Support both ``python -m dohc2_c4lib...`` and direct execution of this file.
# In the latter case Python only adds the package directory to sys.path, while
# importing ``dohc2_c4lib`` requires its parent directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dohc2_c4lib import show_oak4p_fsin_sync as base


WINDOW_NAME = "OAK-4P FSIN Sync Verification - Installation Flip"
HORIZONTAL_FLIP_CAMERAS = frozenset(("CAM_A", "CAM_B", "CAM_C", "CAM_D"))


def flip_sample_for_camera(name: str, sample: base.FrameSample) -> base.FrameSample:
    """Apply the display transform required by one physical camera."""
    socket_name = base.CAMERA_MAPPING[name]
    flip_code = -1 if socket_name in HORIZONTAL_FLIP_CAMERAS else 0
    return replace(sample, image=cv2.flip(sample.image, flip_code))


def make_installation_flip_tile(name, sample, state, tile_size, delta_us):
    return base.make_tile(
        name,
        flip_sample_for_camera(name, sample),
        state,
        tile_size,
        delta_us,
    )


def main(argv=None) -> int:
    print(
        "显示模式：CAM_A/CAM_B/CAM_C/CAM_D 四路画面均上下和左右翻转；"
        "同步统计使用原始设备数据。"
    )
    return base.main(
        argv,
        tile_renderer=make_installation_flip_tile,
        window_name=WINDOW_NAME,
    )


if __name__ == "__main__":
    raise SystemExit(main())
