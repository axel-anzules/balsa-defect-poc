from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from balsa_vision.core.types import PanelResult


class JsonlWriter:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, item: PanelResult) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")