# dataset_analysis/roboflow_source_leakage.py
"""
Agrupa las imágenes del dataset por su archivo fuente de Roboflow,
usando el patrón determinístico de nomenclatura "<original>.rf.<hash>.jpg",
y verifica si algún grupo (= mismo panel físico fuente) tiene imágenes
repartidas entre los splits train/val.

A diferencia del enfoque por perceptual hashing (probabilístico, requiere
calibrar un umbral), esta técnica es exacta: el sufijo .rf.<hash> es la
convención documentada de exportación de Roboflow, por lo que agrupa con
certeza total las variantes (original + augmentadas) de una misma imagen
fuente.

Las imágenes que NO siguen el patrón .rf.<hash> (por ejemplo, las que
tienen el patrón de captura directa "YYYYMMDD_HHMMSS_NNN.jpg") se tratan
como fuentes propias de un único elemento, y se reportan aparte como un
subgrupo distinto del dataset (posible origen: capturas añadidas
directamente a Ultralytics HUB sin pasar por Roboflow).

Uso:
    python -m dataset_analysis.roboflow_source_leakage \
        --root-dir /ruta/al/dataset \
        --ndjson /ruta/al/archivo.ndjson \
        --output-dir ./reports
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from balsa_vision.dataset_analysis.common import DatasetInfo, load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# Patrón de exportación de Roboflow: <nombre_original>.rf.<hash_hex>.<ext>
ROBOFLOW_PATTERN = re.compile(r"^(.+)\.rf\.[0-9a-fA-F]+\.(jpg|jpeg|png)$", re.IGNORECASE)

# Patrón de captura directa observado: YYYYMMDD_HHMMSS_NNN.jpg
DIRECT_CAPTURE_PATTERN = re.compile(r"^(\d{8}_\d{6})_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def extract_source_key(filename: str) -> tuple[str, str]:
    """
    Devuelve (source_key, origin_type) para un nombre de archivo.

    origin_type:
        "roboflow"        -> variante de Roboflow, agrupada por nombre original.
        "direct_capture"  -> captura directa (patrón YYYYMMDD_HHMMSS_NNN),
                              cada imagen es su propia fuente (no se agrupa,
                              ya que confirmado por el usuario que cada
                              archivo = panel físico distinto).
        "unknown"         -> no coincide con ningún patrón conocido; se
                              reporta aparte, nunca se asume agrupación.
    """
    rf_match = ROBOFLOW_PATTERN.match(filename)
    if rf_match:
        return rf_match.group(1), "roboflow"

    direct_match = DIRECT_CAPTURE_PATTERN.match(filename)
    if direct_match:
        return filename, "direct_capture"

    return filename, "unknown"


def build_source_groups(dataset: DatasetInfo) -> dict[str, list[tuple[str, str]]]:
    """
    Agrupa imágenes por source_key. Devuelve {source_key: [(path, split), ...]}.
    Solo se agrupan imágenes de origen 'roboflow'; las de 'direct_capture' y
    'unknown' quedan como grupos de tamaño 1 (no se fuerza agrupación sin
    evidencia determinística).
    """
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    origin_counts: dict[str, int] = defaultdict(int)

    for image in dataset.images:
        source_key, origin_type = extract_source_key(image.image_path.name)
        origin_counts[origin_type] += 1
        groups[source_key].append((str(image.image_path), image.split))

    logger.info("Distribución de origen de archivos: %s", dict(origin_counts))
    return groups


def analyze_leakage(groups: dict[str, list[tuple[str, str]]]) -> dict:
    """Determina qué grupos (fuentes Roboflow) tienen imágenes en más de un split."""
    leaking_groups: list[dict] = []
    clean_multi_image_groups = 0
    single_image_groups = 0
    total_images_in_leaking_groups = 0

    for source_key, entries in groups.items():
        if len(entries) == 1:
            single_image_groups += 1
            continue

        splits_present: dict[str, list[str]] = defaultdict(list)
        for path, split in entries:
            splits_present[split].append(path)

        if len(splits_present) > 1:
            leaking_groups.append(
                {
                    "source_key": source_key,
                    "group_size": len(entries),
                    "split_breakdown": {s: len(p) for s, p in splits_present.items()},
                    "files_by_split": dict(splits_present),
                }
            )
            total_images_in_leaking_groups += len(entries)
        else:
            clean_multi_image_groups += 1

    return {
        "total_source_groups": len(groups),
        "single_image_groups": single_image_groups,
        "clean_multi_image_groups": clean_multi_image_groups,
        "leaking_groups_count": len(leaking_groups),
        "total_images_in_leaking_groups": total_images_in_leaking_groups,
        "leaking_groups_detail": leaking_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verifica leakage train/val usando la clave exacta de "
        "nomenclatura de exportación de Roboflow (.rf.<hash>)."
    )
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--ndjson", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./reports"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(root_dir=args.root_dir, ndjson_path=args.ndjson)
    groups = build_source_groups(dataset)
    report = analyze_leakage(groups)

    report_path = args.output_dir / "roboflow_source_leakage_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("=" * 70)
    logger.info("REPORTE DE LEAKAGE POR FUENTE ROBOFLOW (.rf.<hash>)")
    logger.info("=" * 70)
    logger.info("Grupos de fuente totales           : %d", report["total_source_groups"])
    logger.info("Grupos de una sola imagen           : %d", report["single_image_groups"])
    logger.info("Grupos multi-imagen y limpios        : %d", report["clean_multi_image_groups"])
    logger.info("Grupos multi-imagen CON leakage      : %d", report["leaking_groups_count"])
    logger.info(
        "Total de imágenes en grupos con leakage: %d", report["total_images_in_leaking_groups"]
    )
    if report["leaking_groups_count"] > 0:
        logger.warning("-" * 70)
        logger.warning("⚠️  Muestra de grupos con leakage confirmado (máx. 10):")
        for group in report["leaking_groups_detail"][:10]:
            logger.warning(
                "Fuente '%s' (tamaño=%d): %s",
                group["source_key"], group["group_size"], group["split_breakdown"],
            )
        logger.warning("-" * 70)
        logger.warning(
            "⚠️  SE REQUIERE RE-SPLIT: hay paneles cuyas variantes augmentadas "
            "quedaron repartidas entre train y val."
        )
    else:
        logger.info("✅ Sin leakage detectado: cada fuente Roboflow está contenida en un único split.")
    logger.info("Reporte guardado en: %s", report_path)


if __name__ == "__main__":
    main()