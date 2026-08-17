# src/balsa_vision/preprocess/verify_origin_composition.py
"""
Verifica la composición de origen (Roboflow vs. captura directa) del
dataset tras el crop, para informar si el desbalance de
origen es relevante al diseñar el resto del preprocesamiento,
dado que las imágenes de origen Roboflow ya tienen aplicado Auto-Adjust
Contrast y las de captura directa no.

Reutiliza el mismo patrón determinístico de clasificación de
roboflow_source_leakage.py, aplicado ahora sobre el dataset
post-crop en vez del dataset original, para saber qué sobrevivió a la
exclusión.

Uso:
    python -m balsa_vision.preprocess.verify_origin_composition \
        --cropped-root-dir /ruta/al/dataset_yolo_cropped \
        --output-dir ./reports
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

ROBOFLOW_PATTERN = re.compile(r"^(.+)\.rf\.[0-9a-fA-F]+\.(jpg|jpeg|png)$", re.IGNORECASE)
DIRECT_CAPTURE_PATTERN = re.compile(r"^(\d{8}_\d{6})_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def classify_origin(filename: str) -> str:
    """Clasifica un archivo como 'roboflow', 'direct_capture' o 'unknown'."""
    if ROBOFLOW_PATTERN.match(filename):
        return "roboflow"
    if DIRECT_CAPTURE_PATTERN.match(filename):
        return "direct_capture"
    return "unknown"


def analyze_composition(cropped_root_dir: Path) -> dict:
    """
    Recorre images/train y images/val del dataset recortado y cuenta
    cuántas imágenes de cada origen sobrevivieron, tanto en total como
    por split (relevante porque un desbalance de origen concentrado en
    un solo split sería más problemático que uno distribuido parejo).
    """
    counts_by_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_counts: dict[str, int] = defaultdict(int)
    unknown_samples: list[str] = []

    for split in ("train", "val"):
        images_dir = cropped_root_dir / "images" / split
        if not images_dir.exists():
            continue
        for image_path in images_dir.glob("*"):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            origin = classify_origin(image_path.name)
            counts_by_split[split][origin] += 1
            total_counts[origin] += 1
            if origin == "unknown" and len(unknown_samples) < 10:
                unknown_samples.append(image_path.name)

    total_images = sum(total_counts.values())
    return {
        "total_images": total_images,
        "total_counts": dict(total_counts),
        "total_ratios": {
            origin: count / total_images for origin, count in total_counts.items()
        } if total_images else {},
        "counts_by_split": {split: dict(counts) for split, counts in counts_by_split.items()},
        "unknown_samples": unknown_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verifica composición de origen (Roboflow vs. captura directa) "
        "del dataset tras el crop"
    )
    parser.add_argument("--cropped-root-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./reports"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = analyze_composition(args.cropped_root_dir)

    report_path = args.output_dir / "origin_composition_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("=" * 70)
    logger.info("REPORTE DE COMPOSICIÓN DE ORIGEN (post-crop)")
    logger.info("=" * 70)
    logger.info("Total de imágenes         : %d", report["total_images"])
    logger.info("Conteo por origen         : %s", report["total_counts"])
    logger.info("Proporción por origen     : %s",
                {k: f"{v:.1%}" for k, v in report["total_ratios"].items()})
    logger.info("Desglose por split        : %s", report["counts_by_split"])
    if report["unknown_samples"]:
        logger.warning("Archivos con origen desconocido (muestra): %s", report["unknown_samples"])
    logger.info("Reporte guardado en: %s", report_path)


if __name__ == "__main__":
    main()