# dataset_analysis/duplicates_report.py
"""
Detección de imágenes duplicadas o casi-duplicadas mediante perceptual
hashing (phash), relevante para identificar frames de la misma ráfaga de
captura con variación mínima que podrían sobrerrepresentar un caso
particular en el dataset.

Requiere: pip install imagehash pillow

Uso:
    python -m dataset_analysis.duplicates_report \
        --root-dir /ruta/al/dataset \
        --output-dir ./reports \
        --hash-distance-threshold 5
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image

from scripts.dataset_analysis.common import DatasetInfo, load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


def compute_phashes(dataset: DatasetInfo) -> dict[str, imagehash.ImageHash]:
    """Calcula el perceptual hash de cada imagen del dataset."""
    hashes: dict[str, imagehash.ImageHash] = {}
    for image in dataset.images:
        try:
            with Image.open(image.image_path) as img:
                hashes[str(image.image_path)] = imagehash.phash(img)
        except Exception as exc:  # noqa: BLE001
            logger.warning("No se pudo procesar %s: %s", image.image_path, exc)
    return hashes


def find_near_duplicates(
    hashes: dict[str, imagehash.ImageHash], distance_threshold: int
) -> list[list[str]]:
    """
    Agrupa imágenes cuya distancia de Hamming entre phashes es menor o
    igual al umbral, indicando alta similitud visual.

    Nota de complejidad: esta implementación es O(n^2) sobre el número de
    imágenes, aceptable para datasets del orden de miles (como el actual,
    ~1900 imágenes toma segundos), pero debe revisarse si el dataset
    crece a decenas de miles de imágenes (requeriría LSH o similar).
    """
    items = list(hashes.items())
    visited: set[int] = set()
    groups: list[list[str]] = []

    for i in range(len(items)):
        if i in visited:
            continue
        path_i, hash_i = items[i]
        group = [path_i]
        for j in range(i + 1, len(items)):
            if j in visited:
                continue
            path_j, hash_j = items[j]
            if hash_i - hash_j <= distance_threshold:
                group.append(path_j)
                visited.add(j)
        if len(group) > 1:
            groups.append(group)
            visited.add(i)

    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Reporte de detección de duplicados/near-duplicates.")
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--ndjson", type=Path, default=None,
                        help="Ruta al .ndjson original (fuente recomendada para class_names).")
    parser.add_argument("--data-yaml", type=Path, default=None,
                        help="Alternativa a --ndjson si ya cuentas con data.yaml estándar.")
    parser.add_argument("--output-dir", type=Path, default=Path("./reports"))
    parser.add_argument("--hash-distance-threshold", type=int, default=5,
                         help="Distancia de Hamming máxima para considerar duplicado (0=idéntico).")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(root_dir=args.root_dir, ndjson_path=args.ndjson, data_yaml_path=args.data_yaml)

    logger.info("Calculando perceptual hashes para %d imágenes...", len(dataset.images))
    hashes = compute_phashes(dataset)

    logger.info("Buscando grupos de near-duplicates (umbral=%d)...", args.hash_distance_threshold)
    groups = find_near_duplicates(hashes, args.hash_distance_threshold)

    total_images_in_groups = sum(len(g) for g in groups)
    report = {
        "total_images_analyzed": len(hashes),
        "duplicate_groups_found": len(groups),
        "total_images_involved_in_duplicates": total_images_in_groups,
        "duplicate_ratio": total_images_in_groups / len(hashes) if hashes else 0.0,
        "groups": groups,
    }

    report_path = args.output_dir / "duplicates_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("=" * 70)
    logger.info("REPORTE DE DUPLICADOS / NEAR-DUPLICATES")
    logger.info("=" * 70)
    logger.info("Imágenes analizadas          : %d", report["total_images_analyzed"])
    logger.info("Grupos de duplicados hallados : %d", report["duplicate_groups_found"])
    logger.info("Imágenes involucradas         : %d (%.1f%% del dataset)",
                report["total_images_involved_in_duplicates"], report["duplicate_ratio"] * 100)
    logger.info("Reporte guardado en: %s", report_path)


if __name__ == "__main__":
    main()