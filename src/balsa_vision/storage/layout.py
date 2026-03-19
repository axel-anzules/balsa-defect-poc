from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class ResultsLayout:
    root: Path
    raw_dir: Path
    annotated_dir: Path
    logs_dir: Path


def make_results_layout(results_root: str) -> ResultsLayout:
    d = date.today().isoformat()  # YYYY-MM-DD
    root = Path(results_root) / d
    raw_dir = root / "raw"
    annotated_dir = root / "annotated"
    logs_dir = root / "logs"

    raw_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return ResultsLayout(root=root, raw_dir=raw_dir, annotated_dir=annotated_dir, logs_dir=logs_dir)