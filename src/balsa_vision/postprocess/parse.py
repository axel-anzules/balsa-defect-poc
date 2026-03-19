from __future__ import annotations

from typing import List, Sequence
import numpy as np

from balsa_vision.core.types import Detection
from balsa_vision.preprocess.transforms import LetterboxMeta

def _clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def undo_letterbox_xyxy(xyxy: tuple[float, float, float, float], meta: LetterboxMeta) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    # eliminacion de padding
    x1 = (x1 - meta.pad_w)/meta.scale
    x2 = (x2 - meta.pad_w)/meta.scale
    y1 = (y1 - meta.pad_h)/meta.scale
    y2 = (y2 - meta.pad_h)/meta.scale

    x1i = _clip(int(round(x1)), 0, meta.orig_w - 1)
    x2i = _clip(int(round(x2)), 0, meta.orig_w - 1)
    y1i = _clip(int(round(y1)), 0, meta.orig_h - 1)
    y2i = _clip(int(round(y2)), 0, meta.orig_h - 1)

    return x1i, y1i, x2i, y2i

def parse_simple_out(
        outputs: List[np.ndarray],
        meta: LetterboxMeta,
        class_names: Sequence[str],
        conf_thres: float,
) -> List[Detection]:
    """Parser minimo para output (1, N, 6): x1, y1, x2, y2, score, class_id"""
    arr = outputs[0]
    if arr.ndim != 3 or arr.shape[0] !=1 or arr.shape[2] != 6:
        raise ValueError(
            f"Unsopported ONNX output shape {arr.shape}."
            "Expected (1, N, 6) we adapt parser once we se your actual output"
        )
    
    dets: List[Detection] = []
    for row in arr[0]:
        x1, y1, x2, y2, score, cls_id = row.tolist()
        if float(score) < conf_thres:
            continue
        ci = int(cls_id)
        name = class_names[ci] if 0 <= ci <= len(class_names) else f"class_{ci}"
        xyxy_orig = undo_letterbox_xyxy((x1, y1, x2, y2), meta)
        dets.append(Detection(cls_name=name, score=float(score), xyxy=xyxy_orig))
    return dets