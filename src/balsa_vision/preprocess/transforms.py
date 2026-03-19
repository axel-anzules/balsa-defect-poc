from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import cv2
import numpy as np

@dataclass(frozen=True)
class LetterboxMeta:
    """Metadata para deshacer letterbox"""
    scale: float
    pad_w: int
    pad_h: int
    new_w: int
    new_h: int
    orig_w: int
    orig_h: int

def letterbox_bgr(image: np.ndarray, new_size: int) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize con padding manteniendo aspect ratio (estilo YOLO)."""
    orig_h, orig_w = image.shape[:2]
    scale = min(new_size/orig_w, new_size/orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = cv2.resize(image, (new_w,new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = new_size - new_w
    pad_h = new_size - new_h
    left = pad_w//2
    top = pad_h//2

    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized

    meta = LetterboxMeta(
        scale=scale,
        pad_w=left,
        pad_h=top,
        new_w=new_w,
        new_h=new_h,
        orig_w=orig_w,
        orig_h=orig_h,
    )
    return canvas, meta

def bgr_to_nchw_float(image: np.ndarray) -> np.ndarray:
    """BGR uint8 HWC -> float32 NCHW, rango 0...1."""
    x = image.astype(np.float32)/255.0
    x = np.transpose(x, (2, 0, 1)) # CHW
    x = np.expand_dims(x, axis=0) # NCHW
    return x
