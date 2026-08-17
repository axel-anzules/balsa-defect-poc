from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RESULTS_CSV_HEADER = [
    "timestamp", "experiment_id", "model", "rect", "clahe",
    "augmentation", "precision", "recall", "map50", "map50_95",
    "map50_95_nudillo_nudohueco", "map50_95_sheck_ecuatoriano",
    "train_time_minutes", "run_dir",
]

def log_experiment_result(
        results_csv_path: Path,
        experiment_id: str,
        config: dict[str, Any],
        metrics: dict[str, Any],
        run_dir: str,
) -> None:
    file_exists = results_csv_path.exists()
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with results_csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "experiment_id": experiment_id,
                "model": config.get("model"),
                "imgsz": config.get("imgsz"),
                "rect": config.get("rect"),
                "clahe": config.get("clahe"),
                "augmentation": config.get("augmentation"),
                "precision": metrics.get("recall"),
                "recall": metrics.get("recall"),
                "map50": metrics.get("map50"),
                "map50_95": metrics.get("map50_95"),
                "map50_95_nudillo_nudohueco": metrics.get("map50_95_nudillo_nudohueco"),
                "map50_95_sheck_ecuatoriano": metrics.get("map50_95_sheck_ecuatoriano"),
                "train_time_minutes": metrics.get("train_time_minutes"),
                "run_dir": run_dir,
            }
        )
    logger.info("Resultado de %s registrado en %s", experiment_id, results_csv_path)