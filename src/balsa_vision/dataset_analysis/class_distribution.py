# dataset_analysis/class_distribution.py
"""
Reporte de distribución de clases: instancias por clase, imágenes por
clase, ratio de negativos, índice de desbalance.

Distingue explícitamente "instancias" (cada bounding box individual) de
"imágenes" (una imagen puede contener múltiples instancias de la misma o
distintas clases) porque confundir ambas conduce a conclusiones erróneas
sobre el desbalance real del dataset.

Uso:
    python -m dataset_analysis.class_distribution \
        --root-dir /ruta/al/dataset \
        --output-dir ./reports
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from balsa_vision.dataset_analysis.common import DatasetInfo, load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


def compute_class_distribution(dataset: DatasetInfo) -> dict:
    """Calcula todas las métricas de distribución de clases del dataset."""
    instance_counts: Counter[str] = Counter()
    image_counts: Counter[str] = Counter()
    instance_counts_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    negatives_by_split: Counter[str] = Counter()
    total_by_split: Counter[str] = Counter()

    for image in dataset.images:
        total_by_split[image.split] += 1
        if image.is_negative:
            negatives_by_split[image.split] += 1
            continue

        classes_in_this_image: set[int] = set()
        for box in image.boxes:
            class_name = dataset.class_names[box.class_id]
            instance_counts[class_name] += 1
            instance_counts_by_split[image.split][class_name] += 1
            classes_in_this_image.add(box.class_id)

        for class_id in classes_in_this_image:
            image_counts[dataset.class_names[class_id]] += 1

    # Índice de desbalance: ratio entre la clase más frecuente y la menos
    # frecuente, medido sobre CONTEO DE IMÁGENES (más representativo del
    # muestreo real durante entrenamiento que el conteo de instancias).
    if image_counts:
        max_count = max(image_counts.values())
        min_count = min(image_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")
    else:
        imbalance_ratio = None

    total_images = len(dataset.images)
    total_negatives = sum(negatives_by_split.values())

    return {
        "total_images": total_images,
        "total_negatives": total_negatives,
        "negative_ratio": total_negatives / total_images if total_images else 0.0,
        "instance_counts": dict(instance_counts),
        "image_counts": dict(image_counts),
        "instance_counts_by_split": {
            split: dict(counts) for split, counts in instance_counts_by_split.items()
        },
        "negatives_by_split": dict(negatives_by_split),
        "total_by_split": dict(total_by_split),
        "imbalance_ratio_by_images": imbalance_ratio,
    }


def plot_distribution(report: dict, output_dir: Path) -> None:
    """Genera gráficos de barras comparando instancias vs. imágenes por clase."""
    class_names = list(report["instance_counts"].keys())
    instance_values = [report["instance_counts"][c] for c in class_names]
    image_values = [report["image_counts"][c] for c in class_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(class_names))
    width = 0.35
    ax.bar([i - width / 2 for i in x], instance_values, width, label="Instancias (bboxes)")
    ax.bar([i + width / 2 for i in x], image_values, width, label="Imágenes")
    ax.set_xticks(list(x))
    ax.set_xticklabels(class_names, rotation=15)
    ax.set_ylabel("Cantidad")
    ax.set_title("Distribución de clases: instancias vs. imágenes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    # Gráfico de composición del dataset (positivos por clase vs negativos)
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = class_names + ["Negativos (sin defecto)"]
    values = image_values + [report["total_negatives"]]
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Composición del dataset por imágenes")
    fig.tight_layout()
    fig.savefig(output_dir / "dataset_composition.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reporte de distribución de clases.")
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--ndjson", type=Path, default=None,
                        help="Ruta al .ndjson original (fuente recomendada para class_names).")
    parser.add_argument("--data-yaml", type=Path, default=None,
                        help="Alternativa a --ndjson si ya cuentas con data.yaml estándar.")
    parser.add_argument("--output-dir", type=Path, default=Path("./reports"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(root_dir=args.root_dir, ndjson_path=args.ndjson, data_yaml_path=args.data_yaml)
    report = compute_class_distribution(dataset)

    report_path = args.output_dir / "class_distribution_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_distribution(report, args.output_dir)

    logger.info("=" * 70)
    logger.info("REPORTE DE DISTRIBUCIÓN DE CLASES")
    logger.info("=" * 70)
    logger.info("Total imágenes         : %d", report["total_images"])
    logger.info("Total negativos         : %d (%.1f%%)",
                report["total_negatives"], report["negative_ratio"] * 100)
    logger.info("Instancias por clase    : %s", report["instance_counts"])
    logger.info("Imágenes por clase      : %s", report["image_counts"])
    logger.info("Distribución por split  : %s", report["total_by_split"])
    logger.info("Índice de desbalance    : %.2fx (imagen más común / menos común)",
                report["imbalance_ratio_by_images"])
    logger.info("Reporte guardado en: %s", report_path)
    logger.info("Gráficos guardados en: %s", args.output_dir)


if __name__ == "__main__":
    main()