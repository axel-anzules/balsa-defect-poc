# src/balsa_vision/preprocess/clahe_calibration.py
"""
Utilidad de calibración visual: aplica CLAHE con distintas combinaciones
de clipLimit / tileGridSize sobre una muestra estratificada por brillo
del dataset recortado (post Fase 2.1), para calibrar empíricamente los
parámetros antes de fijarlos en el pipeline de producción (ver Fase 2.2).

CLAHE se aplica sobre el canal L (luminancia) del espacio Lab, no
directamente sobre BGR, para no distorsionar el balance de color del
panel — relevante porque el Hue del panel es la señal usada en la
detección de Fase 2.1 (ADR-015) y no debe alterarse innecesariamente.

Para cada imagen de muestra se generan dos vistas:
    - Panel completo: original vs. cada combinación de parámetros.
    - Zoom sobre una región con defecto (si la imagen tiene anotaciones):
      permite juzgar el efecto de CLAHE justo donde más importa.

No decide nada por sí solo: es una utilidad de inspección visual previa
a fijar CLAHE_CLIP_LIMIT / CLAHE_TILE_GRID_SIZE en el pipeline final.

Uso:
    python -m balsa_vision.preprocess.clahe_calibration \
        --root-dir /ruta/al/dataset_yolo_cropped \
        --ndjson /ruta/al/archivo.ndjson \
        --output-dir ./reports/clahe_calibration \
        --n-per-stratum 3
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from balsa_vision.dataset_analysis.common import DatasetImage, load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# Grilla de combinaciones a comparar. clipLimit bajo (1.0-2.0) produce
# un efecto sutil; valores altos (4.0+) pueden sobre-amplificar ruido
# local. tileGridSize pequeño (4,4) da ecualización muy local; grande
# (16,16) se acerca al comportamiento de ecualización global.
CLAHE_PARAM_GRID: list[tuple[float, tuple[int, int]]] = [
    (1.5, (8, 8)),
    (2.5, (8, 8)),
    (2.5, (4, 4)),
    (4.0, (8, 8)),
]


def apply_clahe(image_bgr: np.ndarray, clip_limit: float, tile_grid_size: tuple[int, int]) -> np.ndarray:
    """Aplica CLAHE sobre el canal L de Lab y reconstruye la imagen BGR."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_equalized = clahe.apply(l_channel)

    lab_equalized = cv2.merge([l_equalized, a_channel, b_channel])
    return cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)


def compute_mean_brightness(image_bgr: np.ndarray) -> float:
    """Brillo medio (canal L de Lab), usado para estratificar la muestra."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    return float(lab[:, :, 0].mean())


def get_defect_crop(image_bgr: np.ndarray, dataset_image: DatasetImage, margin_px: int = 40) -> np.ndarray | None:
    """
    Extrae un recorte centrado en el primer defecto anotado de la
    imagen, con margen, para inspección de detalle. Devuelve None si la
    imagen no tiene anotaciones (negativo).
    """
    if not dataset_image.boxes:
        return None
    box = dataset_image.boxes[0]
    h, w = image_bgr.shape[:2]
    cx, cy = int(box.x_center * w), int(box.y_center * h)
    box_w, box_h = int(box.width * w), int(box.height * h)

    half_side = max(box_w, box_h) // 2 + margin_px
    x1 = max(0, cx - half_side)
    y1 = max(0, cy - half_side)
    x2 = min(w, cx + half_side)
    y2 = min(h, cy + half_side)
    return image_bgr[y1:y2, x1:x2]


def select_stratified_sample(dataset_images: list[DatasetImage], n_per_stratum: int, seed: int) -> dict[str, list[DatasetImage]]:
    """
    Estratifica la muestra por brillo medio (oscura/media/clara) según
    los propios terciles del dataset, priorizando imágenes CON defecto
    anotado (para poder generar el zoom de detalle).
    """
    random.seed(seed)
    with_defects = [img for img in dataset_images if img.boxes]

    scored: list[tuple[float, DatasetImage]] = []
    for img in with_defects:
        image_bgr = cv2.imread(str(img.image_path))
        if image_bgr is None:
            continue
        scored.append((compute_mean_brightness(image_bgr), img))

    scored.sort(key=lambda x: x[0])
    n = len(scored)
    strata = {
        "oscura": [item[1] for item in scored[: n // 3]],
        "media": [item[1] for item in scored[n // 3 : 2 * n // 3]],
        "clara": [item[1] for item in scored[2 * n // 3 :]],
    }
    return {
        name: random.sample(images, min(n_per_stratum, len(images)))
        for name, images in strata.items()
    }


def build_comparison_figure(dataset_image: DatasetImage, output_path: Path) -> None:
    """
    Genera una figura por imagen de muestra: fila superior con el panel
    completo (original + cada combinación de CLAHE), fila inferior con
    el zoom sobre el defecto (misma serie de transformaciones).
    """
    image_bgr = cv2.imread(str(dataset_image.image_path))
    if image_bgr is None:
        logger.warning("No se pudo leer %s, se omite.", dataset_image.image_path)
        return

    defect_crop = get_defect_crop(image_bgr, dataset_image)
    n_variants = 1 + len(CLAHE_PARAM_GRID)  # original + combinaciones
    n_rows = 2 if defect_crop is not None else 1

    fig, axes = plt.subplots(n_rows, n_variants, figsize=(4 * n_variants, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    axes[0, 0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original", fontsize=10)
    axes[0, 0].axis("off")
    if defect_crop is not None:
        axes[1, 0].imshow(cv2.cvtColor(defect_crop, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("Original (zoom defecto)", fontsize=10)
        axes[1, 0].axis("off")

    for col, (clip_limit, tile_grid) in enumerate(CLAHE_PARAM_GRID, start=1):
        equalized_full = apply_clahe(image_bgr, clip_limit, tile_grid)
        axes[0, col].imshow(cv2.cvtColor(equalized_full, cv2.COLOR_BGR2RGB))
        axes[0, col].set_title(f"clip={clip_limit}, tile={tile_grid}", fontsize=10)
        axes[0, col].axis("off")

        if defect_crop is not None:
            equalized_crop = apply_clahe(defect_crop, clip_limit, tile_grid)
            axes[1, col].imshow(cv2.cvtColor(equalized_crop, cv2.COLOR_BGR2RGB))
            axes[1, col].set_title(f"Zoom clip={clip_limit}, tile={tile_grid}", fontsize=10)
            axes[1, col].axis("off")

    fig.suptitle(dataset_image.image_path.name, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibración visual de parámetros CLAHE.")
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--ndjson", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./reports/clahe_calibration"))
    parser.add_argument("--n-per-stratum", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(root_dir=args.root_dir, ndjson_path=args.ndjson)

    strata_samples = select_stratified_sample(dataset.images, args.n_per_stratum, args.seed)

    for stratum_name, images in strata_samples.items():
        for idx, dataset_image in enumerate(images, start=1):
            out_path = args.output_dir / f"clahe_review_{stratum_name}_{idx:02d}.png"
            build_comparison_figure(dataset_image, out_path)
            logger.info("Guardado: %s", out_path)

    logger.info("Calibración visual de CLAHE completa en: %s", args.output_dir)


if __name__ == "__main__":
    main()