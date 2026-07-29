# src/balsa_vision/preprocess/clahe_dataset.py
"""
Orquestador de Fase 2.2: aplica CLAHE (image_enhancement.apply_clahe)
sobre todo el dataset ya recortado (Fase 2.1), materializando el
resultado final en disco.

A diferencia del crop (Fase 2.1), esta es una transformación puramente
fotométrica: no afecta la geometría de las bounding boxes, por lo que
los archivos de label se copian sin modificación.

El dataset resultante (dataset_yolo_preprocessed) es el candidato final
para pasar a Fase 3 (Data Augmentation) y Fase 4 (Entrenamiento).

Uso:
    python -m balsa_vision.preprocess.clahe_dataset \
        --input-dir /ruta/al/dataset_yolo_cropped \
        --output-dir /ruta/al/dataset_yolo_preprocessed
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import cv2
import yaml

from balsa_vision.preprocess.image_enhancement import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    apply_clahe,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


def process_split(input_dir: Path, output_dir: Path, split: str) -> dict:
    """
    Procesa un split completo (train o val): aplica CLAHE a cada imagen
    y copia su label correspondiente sin modificar.
    """
    images_in_dir = input_dir / "images" / split
    labels_in_dir = input_dir / "labels" / split
    images_out_dir = output_dir / "images" / split
    labels_out_dir = output_dir / "labels" / split
    images_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)

    if not images_in_dir.exists():
        logger.warning("No existe %s, se omite split '%s'.", images_in_dir, split)
        return {"total": 0, "processed": 0, "failed": []}

    image_paths = sorted(
        p for p in images_in_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    processed_count = 0
    failed: list[str] = []

    for i, image_path in enumerate(image_paths, start=1):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            failed.append(str(image_path))
            logger.warning("No se pudo leer %s, se omite.", image_path)
            continue

        enhanced = apply_clahe(image_bgr)
        cv2.imwrite(str(images_out_dir / image_path.name), enhanced)

        label_path = labels_in_dir / f"{image_path.stem}.txt"
        dest_label_path = labels_out_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, dest_label_path)
        else:
            # Negativo intencional (sin defectos), consistente con ADR-001.
            dest_label_path.write_text("", encoding="utf-8")

        processed_count += 1
        if i % 200 == 0:
            logger.info("Split '%s': %d/%d", split, i, len(image_paths))

    return {"total": len(image_paths), "processed": processed_count, "failed": failed}


def write_data_yaml(output_dir: Path, source_data_yaml: Path) -> None:
    """Copia data.yaml del dataset origen, actualizando la ruta 'path'."""
    with source_data_yaml.open("r", encoding="utf-8") as f:
        data_yaml = yaml.safe_load(f)
    data_yaml["path"] = str(output_dir.resolve())
    with (output_dir / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aplica CLAHE calibrado (ADR-018) sobre el dataset "
        "recortado (Fase 2.1), materializando el resultado en disco."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = {"clip_limit": CLAHE_CLIP_LIMIT, "tile_grid_size": CLAHE_TILE_GRID_SIZE, "splits": {}}

    for split in ("train", "val"):
        logger.info("Procesando split '%s'...", split)
        split_report = process_split(args.input_dir, args.output_dir, split)
        report["splits"][split] = split_report
        logger.info(
            "Split '%s' completo: %d/%d procesadas, %d fallidas.",
            split, split_report["processed"], split_report["total"], len(split_report["failed"]),
        )

    source_data_yaml = args.input_dir / "data.yaml"
    if source_data_yaml.exists():
        write_data_yaml(args.output_dir, source_data_yaml)
    else:
        logger.warning("No se encontró data.yaml en %s; no se generó uno nuevo.", args.input_dir)

    report_path = args.output_dir / "clahe_processing_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    total_processed = sum(s["processed"] for s in report["splits"].values())
    total_images = sum(s["total"] for s in report["splits"].values())
    logger.info("=" * 70)
    logger.info("CLAHE aplicado: %d/%d imágenes procesadas exitosamente.", total_processed, total_images)
    logger.info("Dataset final listo en: %s", args.output_dir)
    logger.info("Reporte guardado en: %s", report_path)


if __name__ == "__main__":
    main()