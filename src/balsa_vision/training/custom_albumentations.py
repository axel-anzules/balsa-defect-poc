from __future__ import annotations
import logging
import albumentations as A
logger = logging.getLogger(__name__)

APPROVED_TRANSFORMS: list[A.BasicTransform] = [
    A.MotionBlur(
        blur_limit=(3, 7),
        angle_range=(0, 15),
        direction_range=(-0.3, 0.3),
        p=0.08,
    ),
    A.CoarseDropout(
        num_holes_range=(1, 3),
        hole_height_range=(0.02, 0.06),
        hole_width_range=(0.02, 0.06),
        p=0.15,
    ),
    A.GaussNoise(std_range=(0.02, 0.06), p=0.10),
]