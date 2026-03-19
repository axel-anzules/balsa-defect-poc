from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol
import numpy as np


@dataclass(frozen=True)
class Frame:
    """
    Representa un frame capturado desde cualquier fuente.

    image: imagen BGR (OpenCV) uint8 [H,W,3]
    frame_id: identificador del frame o nombre de archivo
    timestamp_ms: tiempo en milisegundos
    """
    image: np.ndarray
    frame_id: str
    timestamp_ms: int


class FrameSource(Protocol):
    """
    Interfaz para cualquier fuente de frames.
    Implementaciones:
    - FolderSource
    - CameraSource
    """

    def __iter__(self) -> Iterator[Frame]:
        ...

    def close(self) -> None:
        ...