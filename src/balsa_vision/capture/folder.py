from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional
import time

import cv2
import numpy as np

from balsa_vision.capture.base import Frame


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class FolderSource:
    """Itera imágenes de una carpeta como frames."""
    folder: str
    recursive: bool = False
    sort: bool = True

    def __iter__(self) -> Iterator[Frame]:
        paths = self._collect_paths()
        for p in paths:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            ts_ms = int(time.time() * 1000)
            yield Frame(image=img, frame_id=p.stem, timestamp_ms=ts_ms)

    def close(self) -> None:
        return

    def _collect_paths(self) -> List[Path]:
        root = Path(self.folder)
        if not root.exists():
            raise FileNotFoundError(f"Folder not found: {root}")

        if self.recursive:
            paths = [p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
        else:
            paths = [p for p in root.glob("*") if p.suffix.lower() in SUPPORTED_EXTS]

        if self.sort:
            paths.sort(key=lambda x: x.name)
        return paths