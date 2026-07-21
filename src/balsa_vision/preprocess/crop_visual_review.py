# src/balsa_vision/preprocess/crop_visual_review.py
"""
Genera una muestra visual estratificada del resultado de panel_crop_dataset.py
para verificación humana antes de aceptar el pipeline de Fase 2.1.

Para cada imagen de muestra, genera un panel comparativo:
    - Izquierda: imagen ORIGINAL con el rectángulo detectado (minAreaRect)
      dibujado en rojo, para juzgar si el algoritmo capturó solo el panel
      o también el marco de espuma.
    - Derecha: imagen RECORTADA final con las bounding boxes reproyectadas
      en verde, para juzgar la calidad del resultado terminado.

La muestra se estratifica por area_ratio (bajo/medio/alto dentro del
rango calibrado) para detectar si el problema de sobre-detección (si
existe) es sistemático o afecta solo a un subconjunto de imágenes.

Uso:
    python -m balsa_vision.preprocess.crop_visual_review \
        --original-root-dir /ruta/al/dataset_yolo_resplit \
        --cropped-root-dir /ruta/al/dataset_yolo_cropped \
        --crop-report /ruta/al/dataset_yolo_cropped/panel_crop_report.json \
        --ndjson /ruta/al/archivo.ndjson \
        --output-dir ./reports/crop_review \
        --n-per-stratum 4
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import cv2
from dotenv import parser
import matplotlib.pyplot as plt
import numpy as np

from balsa_vision.dataset_analysis.common import load_dataset
from balsa_vision.preprocess.panel_cropper import (
    DEFAULT_HUE_RANGE,
    DEFAULT_MIN_SATURATION,
    detect_panel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


def draw_detected_rect(image: np.ndarray, center: tuple, size: tuple, angle: float) -> np.ndarray:
    """Dibuja el minAreaRect detectado sobre una copia de la imagen original."""
    annotated = image.copy()
    box_points = cv2.boxPoints((center, size, angle))
    box_points = np.intp(box_points)
    cv2.drawContours(annotated, [box_points], 0, (0, 0, 255), 4)  # rojo BGR
    return annotated


def draw_cropped_with_boxes(cropped_path: Path, label_path: Path, class_names: dict[int, str]) -> np.ndarray:
    """Dibuja las bounding boxes reproyectadas sobre la imagen ya recortada."""
    img = cv2.imread(str(cropped_path))
    if img is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    h, w = img.shape[:2]
    if label_path.exists():
        content = label_path.read_text(encoding="utf-8").strip()
        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, xc, yc, bw, bh = (float(p) for p in parts)
            class_id = int(class_id)
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, class_names.get(class_id, str(class_id)), (x1, max(y1 - 8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Revisión visual estratificada del crop de panel.")
    parser.add_argument("--original-root-dir", type=Path, required=True)
    parser.add_argument("--cropped-root-dir", type=Path, required=True)
    parser.add_argument("--crop-report", type=Path, required=True)
    parser.add_argument("--ndjson", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./reports/crop_review"))
    parser.add_argument("--n-per-stratum", type=int, default=4)
    parser.add_argument("--hue-min", type=int, default=DEFAULT_HUE_RANGE[0])
    parser.add_argument("--hue-max", type=int, default=DEFAULT_HUE_RANGE[1])
    parser.add_argument("--min-saturation", type=int, default=DEFAULT_MIN_SATURATION)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(root_dir=args.original_root_dir, ndjson_path=args.ndjson)

    with args.crop_report.open("r", encoding="utf-8") as f:
        crop_report = json.load(f)
    excluded_files = {e["file"] for e in crop_report["excluded_detail"]}

    # Re-detectar (barato, solo para esta muestra) sobre las imágenes procesadas
    # exitosamente, para poder dibujar el rectángulo original y clasificar por
    # estrato de area_ratio.
    candidates: list[tuple] = []
    for image in dataset.images:
        if str(image.image_path) in excluded_files:
            continue
        img = cv2.imread(str(image.image_path))
        if img is None:
            continue
        result = detect_panel(img, hue_range=(args.hue_min, args.hue_max), min_saturation=args.min_saturation)
        if result.success:
            candidates.append((image, result.area_ratio, result.panel_rect))

    candidates.sort(key=lambda c: c[1])
    n = len(candidates)
    strata = {
        "area_baja": candidates[: n // 3],
        "area_media": candidates[n // 3 : 2 * n // 3],
        "area_alta": candidates[2 * n // 3 :],
    }

    for stratum_name, items in strata.items():
        sample = random.sample(items, min(args.n_per_stratum, len(items)))
        fig, axes = plt.subplots(len(sample), 2, figsize=(12, 5 * len(sample)))
        if len(sample) == 1:
            axes = axes.reshape(1, 2)

        for row, (image, area_ratio, panel_rect) in enumerate(sample):
            original_img = cv2.imread(str(image.image_path))
            annotated_original = draw_detected_rect(
                original_img, panel_rect.center, panel_rect.size, panel_rect.angle
            )

            cropped_path = args.cropped_root_dir / "images" / image.split / image.image_path.name
            label_path = args.cropped_root_dir / "labels" / image.split / f"{image.image_path.stem}.txt"
            cropped_with_boxes = draw_cropped_with_boxes(cropped_path, label_path, dataset.class_names)

            axes[row, 0].imshow(cv2.cvtColor(annotated_original, cv2.COLOR_BGR2RGB))
            axes[row, 0].set_title(f"Original + rect detectado\narea_ratio={area_ratio:.3f}", fontsize=9)
            axes[row, 0].axis("off")

            axes[row, 1].imshow(cv2.cvtColor(cropped_with_boxes, cv2.COLOR_BGR2RGB))
            axes[row, 1].set_title(f"Crop final + boxes\n{image.image_path.name}", fontsize=9)
            axes[row, 1].axis("off")

        fig.tight_layout()
        out_path = args.output_dir / f"crop_review_{stratum_name}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        logger.info("Guardado: %s (%d muestras)", out_path, len(sample))

    logger.info("Revisión visual completa en: %s", args.output_dir)


if __name__ == "__main__":
    main()