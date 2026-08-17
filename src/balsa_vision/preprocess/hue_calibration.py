# src/balsa_vision/preprocess/hue_calibration.py
"""
Utilidad de calibración: compara la distribución de Hue (matiz) entre la
región central del frame (donde el panel está casi siempre presente) 
y la región periférica (donde predomina el marco de espuma), 
para calibrar empíricamente el rango de Hue que discrimina panel de marco.

No decide nada por sí solo: genera el histograma para inspección visual
humana antes de fijar DEFAULT_HUE_RANGE en panel_cropper.py.

Uso:
    python -m balsa_vision.preprocess.hue_calibration \
        --root-dir /ruta/al/dataset_yolo_resplit \
        --ndjson /ruta/al/archivo.ndjson \
        --output-dir ./reports/hue_calibration \
        --n-samples 30
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from balsa_vision.dataset_analysis.common import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


def collect_hue_samples(image_paths: list[Path], n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Para una muestra de imágenes, recolecta los valores de Hue de:
        - región central (40%-60% de w y h): predominantemente panel.
        - región periférica (borde 10% del frame en cada lado): predominantemente marco/fondo.
    Solo se incluyen píxeles con saturación > 15, para excluir negro puro
    (evita contaminar el histograma con Hue indefinido de píxeles sin color).
    """
    random.seed(seed)
    sample_paths = random.sample(image_paths, min(n_samples, len(image_paths)))

    central_hues: list[np.ndarray] = []
    peripheral_hues: list[np.ndarray] = []

    for path in sample_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue, sat = hsv[:, :, 0], hsv[:, :, 1]

        cy1, cy2 = int(h * 0.4), int(h * 0.6)
        cx1, cx2 = int(w * 0.4), int(w * 0.6)
        central_region_hue = hue[cy1:cy2, cx1:cx2]
        central_region_sat = sat[cy1:cy2, cx1:cx2]
        central_hues.append(central_region_hue[central_region_sat > 15])

        edge = int(w * 0.05)
        peripheral_region_hue = np.concatenate([
            hue[: int(h * 0.1), :].ravel(), hue[int(h * 0.9):, :].ravel(),
            hue[:, :edge].ravel(), hue[:, -edge:].ravel(),
        ])
        peripheral_region_sat = np.concatenate([
            sat[: int(h * 0.1), :].ravel(), sat[int(h * 0.9):, :].ravel(),
            sat[:, :edge].ravel(), sat[:, -edge:].ravel(),
        ])
        peripheral_hues.append(peripheral_region_hue[peripheral_region_sat > 15])

    return np.concatenate(central_hues), np.concatenate(peripheral_hues)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibración de rango de Hue panel vs. marco.")
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--ndjson", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./reports/hue_calibration"))
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(root_dir=args.root_dir, ndjson_path=args.ndjson)
    image_paths = [img.image_path for img in dataset.images]

    logger.info("Recolectando muestras de Hue de %d imágenes...", args.n_samples)
    central, peripheral = collect_hue_samples(image_paths, args.n_samples, args.seed)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(central, bins=90, range=(0, 180), alpha=0.6, label="Región central (panel esperado)", color="orange")
    ax.hist(peripheral, bins=90, range=(0, 180), alpha=0.6, label="Región periférica (marco esperado)", color="steelblue")
    ax.set_xlabel("Hue (OpenCV, rango 0-179)")
    ax.set_ylabel("Frecuencia (píxeles)")
    ax.set_title("Distribución de Hue: región central vs. periférica")
    ax.legend()
    fig.tight_layout()
    out_path = args.output_dir / "hue_distribution_comparison.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    logger.info("Percentiles región central   (P10, P50, P90): %s",
                np.percentile(central, [10, 50, 90]))
    logger.info("Percentiles región periférica (P10, P50, P90): %s",
                np.percentile(peripheral, [10, 50, 90]))
    logger.info("Histograma guardado en: %s", out_path)


if __name__ == "__main__":
    main()