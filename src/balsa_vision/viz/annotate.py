from __future__ import annotations

from typing import List

import cv2
import numpy as np

from balsa_vision.core.types import Detection


def draw_detections(image_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Devuelve una copia anotada con bbox+labels."""
    out: np.ndarray = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.cls_name} {det.score:.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return np.asarray(out)