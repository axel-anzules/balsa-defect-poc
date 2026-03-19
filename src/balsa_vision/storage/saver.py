from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from balsa_vision.storage.layout import ResultsLayout


@dataclass
class EvidenceSaver:
    layout: ResultsLayout

    def save_raw(self, filename: str, image_bgr: np.ndarray) -> str:
        path = self.layout.raw_dir / filename
        cv2.imwrite(str(path), image_bgr)
        return str(path)

    def save_annotated(self, filename: str, image_bgr: np.ndarray) -> str:
        path = self.layout.annotated_dir / filename
        cv2.imwrite(str(path), image_bgr)
        return str(path)