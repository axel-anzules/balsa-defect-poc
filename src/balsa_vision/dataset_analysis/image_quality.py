# dataset_analysis/image_quality.py
"""
Reporte de calidad de imagen: resolución, nitidez (varianza del
Laplaciano) y exposición (brillo medio del histograma en escala de
grises), con detección de imágenes potencialmente problemáticas
(borrosas o mal expuestas) para revisión manual.

Uso:
    python -m dataset_analysis.image_quality \
        --root-dir /ruta/al/dataset \
        --output-dir ./reports \
        --blur-threshold 100.0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from balsa_vision.dataset_analysis.common import DatasetInfo, load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


def compute_sharpness(image_gray: np.ndarray) -> float:
    """
    Varianza del Laplaciano: métrica estándar para detectar blur sin
    modelos de deep learning. Valores bajos indican imagen borrosa
    (bordes poco definidos); el umbral óptimo depende del dominio y debe
    calibrarse visualmente con los casos límite reportados.
    """
    return float(cv2.Laplacian(image_gray, cv2.CV_64F).var())


def compute_brightness(image_gray: np.ndarray) -> float:
    """Brillo medio (0-255) de la imagen en escala de grises."""
    return float(image_gray.mean())


def analyze_image_quality(
    dataset: DatasetInfo, blur_threshold: float, dark_threshold: float, bright_threshold: float
) -> dict:
    """Recorre todas las imágenes calculando métricas de calidad y resolución."""
    resolutions: list[tuple[int, int]] = []
    sharpness_values: list[float] = []
    brightness_values: list[float] = []
    problematic_blurry: list[dict] = []
    problematic_exposure: list[dict] = []
    unreadable: list[str] = []

    for image in dataset.images:
        img = cv2.imread(str(image.image_path))
        if img is None:
            unreadable.append(str(image.image_path))
            continue

        h, w = img.shape[:2]
        resolutions.append((w, h))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = compute_sharpness(gray)
        brightness = compute_brightness(gray)
        sharpness_values.append(sharpness)
        brightness_values.append(brightness)

        if sharpness < blur_threshold:
            problematic_blurry.append(
                {"file": str(image.image_path), "sharpness": sharpness}
            )
        if brightness < dark_threshold or brightness > bright_threshold:
            problematic_exposure.append(
                {"file": str(image.image_path), "brightness": brightness}
            )

    unique_resolutions = sorted(set(resolutions))
    sharpness_arr = np.array(sharpness_values)
    brightness_arr = np.array(brightness_values)

    return {
        "total_analyzed": len(resolutions),
        "unreadable_files": unreadable,
        "unique_resolutions": [{"width": w, "height": h} for w, h in unique_resolutions],
        "resolution_is_consistent": len(unique_resolutions) == 1,
        "sharpness_mean": float(sharpness_arr.mean()) if len(sharpness_arr) else None,
        "sharpness_std": float(sharpness_arr.std()) if len(sharpness_arr) else None,
        "brightness_mean": float(brightness_arr.mean()) if len(brightness_arr) else None,
        "brightness_std": float(brightness_arr.std()) if len(brightness_arr) else None,
        "problematic_blurry_count": len(problematic_blurry),
        "problematic_blurry_sample": sorted(
            problematic_blurry, key=lambda x: x["sharpness"]
        )[:15],
        "problematic_exposure_count": len(problematic_exposure),
        "problematic_exposure_sample": problematic_exposure[:15],
        "_raw_sharpness": sharpness_values,
        "_raw_brightness": brightness_values,
    }


def plot_quality(report: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(report["_raw_sharpness"], bins=40, color="steelblue")
    axes[0].set_title("Distribución de nitidez (Varianza del Laplaciano)")
    axes[0].set_xlabel("Nitidez (mayor = más nítida)")
    axes[0].set_ylabel("Frecuencia")

    axes[1].hist(report["_raw_brightness"], bins=40, color="darkorange")
    axes[1].set_title("Distribución de brillo medio")
    axes[1].set_xlabel("Brillo (0-255)")
    axes[1].set_ylabel("Frecuencia")

    fig.tight_layout()
    fig.savefig(output_dir / "image_quality_distributions.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reporte de calidad de imagen.")
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--ndjson", type=Path, default=None,
                        help="Ruta al .ndjson original (fuente recomendada para class_names).")
    parser.add_argument("--data-yaml", type=Path, default=None,
                        help="Alternativa a --ndjson si ya cuentas con data.yaml estándar.")
    parser.add_argument("--output-dir", type=Path, default=Path("./reports"))
    parser.add_argument("--blur-threshold", type=float, default=100.0,
                         help="Umbral de varianza del Laplaciano bajo el cual se marca como borrosa.")
    parser.add_argument("--dark-threshold", type=float, default=40.0)
    parser.add_argument("--bright-threshold", type=float, default=220.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(root_dir=args.root_dir, ndjson_path=args.ndjson, data_yaml_path=args.data_yaml)
    report = analyze_image_quality(
        dataset, args.blur_threshold, args.dark_threshold, args.bright_threshold
    )

    serializable_report = {k: v for k, v in report.items() if not k.startswith("_raw")}
    report_path = args.output_dir / "image_quality_report.json"
    report_path.write_text(
        json.dumps(serializable_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    plot_quality(report, args.output_dir)

    logger.info("=" * 70)
    logger.info("REPORTE DE CALIDAD DE IMAGEN")
    logger.info("=" * 70)
    logger.info("Imágenes analizadas       : %d", report["total_analyzed"])
    logger.info("Imágenes ilegibles         : %d", len(report["unreadable_files"]))
    logger.info("Resoluciones únicas        : %s", report["unique_resolutions"])
    logger.info("Resolución consistente     : %s", report["resolution_is_consistent"])
    logger.info("Nitidez media ± std        : %.2f ± %.2f",
                report["sharpness_mean"], report["sharpness_std"])
    logger.info("Brillo medio ± std         : %.2f ± %.2f",
                report["brightness_mean"], report["brightness_std"])
    logger.info("Imágenes borrosas (umbral) : %d", report["problematic_blurry_count"])
    logger.info("Imágenes mal expuestas     : %d", report["problematic_exposure_count"])
    logger.info("Reporte guardado en: %s", report_path)


if __name__ == "__main__":
    main()