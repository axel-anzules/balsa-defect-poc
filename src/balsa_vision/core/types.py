from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Detection:
    """Una detección (bbox + clase + score)."""
    cls_name: str
    score: float
    xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass(frozen=True)
class PanelResult:
    """Resultado por panel."""
    panel_id: str
    timestamp_ms: int
    ok: bool
    reason: str
    detections: List[Detection]
    raw_path: str
    annotated_path: str