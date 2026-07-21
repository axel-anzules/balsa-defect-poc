# dataset_analysis/bbox_geometry.py
"""
Reporte de geometría de bounding boxes: distribución de tamaños relativos
al frame completo, aspect ratios, y localización espacial de los defectos
dentro de la imagen.

Este reporte es la base cuantitativa para diseñar la estrategia de
cropping/segmentación del panel en la Fase 2: si los defectos ocupan un
porcentaje muy bajo del área total del frame, cuantifica objetivamente
cuánto "fondo" está presente y da soporte a la decisión de recorte.

Uso:
    python -m dataset_analysis.bbox_geometry \
        --root-dir /ruta/al/dataset \
        --output-dir ./reports
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.dataset_analysis.common import DatasetInfo, load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


def compute_bbox_geometry(dataset: DatasetInfo) -> dict:
    """Extrae estadísticas de área relativa, aspect ratio y posición de todas las bboxes."""
    area_ratios_by_class: dict[str, list[float]] = {name: [] for name in dataset.class_names.values()}
    aspect_ratios_by_class: dict[str, list[float]] = {name: [] for name in dataset.class_names.values()}
    centers_by_class: dict[str, list[tuple[float, float]]] = {
        name: [] for name in dataset.class_names.values()
    }

    for image in dataset.images:
        for box in image.boxes:
            class_name = dataset.class_names[box.class_id]
            area_ratios_by_class[class_name].append(box.area_ratio)
            aspect_ratios_by_class[class_name].append(box.aspect_ratio)
            centers_by_class[class_name].append((box.x_center, box.y_center))

    stats: dict[str, dict] = {}
    for class_name, areas in area_ratios_by_class.items():
        if not areas:
            stats[class_name] = {"count": 0}
            continue
        areas_arr = np.array(areas)
        stats[class_name] = {
            "count": len(areas),
            "area_ratio_mean_pct": float(areas_arr.mean() * 100),
            "area_ratio_median_pct": float(np.median(areas_arr) * 100),
            "area_ratio_min_pct": float(areas_arr.min() * 100),
            "area_ratio_max_pct": float(areas_arr.max() * 100),
            "area_ratio_std_pct": float(areas_arr.std() * 100),
            "aspect_ratio_mean": float(np.mean(aspect_ratios_by_class[class_name])),
        }

    return {
        "per_class_stats": stats,
        "_raw_area_ratios": area_ratios_by_class,  # para plotting, no se serializa
        "_raw_centers": centers_by_class,
    }


def plot_bbox_geometry(geometry: dict, output_dir: Path) -> None:
    """Genera histogramas de área relativa y mapa de calor de posiciones."""
    raw_areas = geometry["_raw_area_ratios"]
    raw_centers = geometry["_raw_centers"]

    # Histograma de área relativa por clase (escala %).
    fig, ax = plt.subplots(figsize=(8, 5))
    for class_name, areas in raw_areas.items():
        if areas:
            ax.hist([a * 100 for a in areas], bins=30, alpha=0.6, label=class_name)
    ax.set_xlabel("Área de la bbox como % del área total de la imagen")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución del tamaño relativo de los defectos")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "bbox_area_ratio_histogram.png", dpi=150)
    plt.close(fig)

    # Mapa de posiciones (dónde caen los defectos dentro del frame).
    fig, axes = plt.subplots(1, len(raw_centers), figsize=(6 * len(raw_centers), 6))
    if len(raw_centers) == 1:
        axes = [axes]
    for ax, (class_name, centers) in zip(axes, raw_centers.items()):
        if centers:
            xs, ys = zip(*centers)
            ax.scatter(xs, [1 - y for y in ys], alpha=0.4, s=15)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Posición de '{class_name}' en el frame")
        ax.set_xlabel("x normalizado")
        ax.set_ylabel("y normalizado (invertido)")
    fig.tight_layout()
    fig.savefig(output_dir / "bbox_spatial_distribution.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reporte de geometría de bounding boxes.")
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--ndjson", type=Path, default=None,
                        help="Ruta al .ndjson original (fuente recomendada para class_names).")
    parser.add_argument("--data-yaml", type=Path, default=None,
                        help="Alternativa a --ndjson si ya cuentas con data.yaml estándar.")
    parser.add_argument("--output-dir", type=Path, default=Path("./reports"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(root_dir=args.root_dir, ndjson_path=args.ndjson, data_yaml_path=args.data_yaml)
    geometry = compute_bbox_geometry(dataset)

    serializable_report = {"per_class_stats": geometry["per_class_stats"]}
    report_path = args.output_dir / "bbox_geometry_report.json"
    report_path.write_text(
        json.dumps(serializable_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    plot_bbox_geometry(geometry, args.output_dir)

    logger.info("=" * 70)
    logger.info("REPORTE DE GEOMETRÍA DE BOUNDING BOXES")
    logger.info("=" * 70)
    for class_name, class_stats in geometry["per_class_stats"].items():
        logger.info("Clase: %s", class_name)
        for key, value in class_stats.items():
            logger.info("    %s: %s", key, value)
    logger.info("Reporte guardado en: %s", report_path)
    logger.info("Gráficos guardados en: %s", args.output_dir)


if __name__ == "__main__":
    main()