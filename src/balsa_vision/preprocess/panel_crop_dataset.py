# preprocess/panel_crop_dataset.py
"""
Orquestador: aplica la detección + enderezado + recorte del
panel (preprocessing/panel_cropper.py) a todo el dataset, en dos pasadas:

    Pasada 1: detecta el panel en todas las imágenes y recolecta la
              distribución real de área_ratio y aspect_ratio, para
              derivar los rangos de validación estadísticamente.
    Pasada 2: aplica la detección + validación + crop + transformación
              de boxes con los rangos calibrados, materializando el
              nuevo dataset. Las imágenes que fallan la validación se
              EXCLUYEN y se registran en un reporte aparte
              para revisión manual.

Uso:
    python -m preprocessing.panel_crop_dataset \
        --root-dir /ruta/al/dataset_yolo_resplit \
        --ndjson /ruta/al/archivo.ndjson \
        --output-dir /ruta/al/dataset_yolo_cropped \
        --margin-ratio 0.06 \
        --percentile-margin 5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import yaml

from balsa_vision.dataset_analysis.common import DatasetInfo, load_dataset
from balsa_vision.preprocess.panel_cropper import (
    crop_panel,
    detect_panel,
    validate_panel_geometry,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


def run_first_pass(dataset: DatasetInfo) -> tuple[dict[str, dict], list[float], list[float]]:
    """
    Detecta el panel en todas las imágenes (sin validar todavía) y
    recolecta las distribuciones de área_ratio y aspect_ratio de los
    paneles detectados exitosamente, para calibrar los rangos de
    validación de la Pasada 2.
    """
    detections: dict[str, dict] = {}
    area_ratios: list[float] = []
    aspect_ratios: list[float] = []

    for i, image in enumerate(dataset.images, start=1):
        img = cv2.imread(str(image.image_path))
        if img is None:
            detections[str(image.image_path)] = {"success": False, "failure_reason": "unreadable_image"}
            continue

        result = detect_panel(img)
        if result.success:
            assert result.panel_rect is not None and result.area_ratio is not None
            area_ratios.append(result.area_ratio)
            aspect_ratios.append(result.panel_rect.aspect_ratio)
            detections[str(image.image_path)] = {
                "success": True,
                "area_ratio": result.area_ratio,
                "aspect_ratio": result.panel_rect.aspect_ratio,
                "center": result.panel_rect.center,
                "size": result.panel_rect.size,
                "angle": result.panel_rect.angle,
            }
        else:
            detections[str(image.image_path)] = {"success": False, "failure_reason": result.failure_reason}

        if i % 200 == 0:
            logger.info("Pasada 1 (detección): %d/%d", i, len(dataset.images))

    return detections, area_ratios, aspect_ratios


def calibrate_validation_ranges(
    area_ratios: list[float], aspect_ratios: list[float], percentile_margin: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Deriva los rangos de validación a partir de percentiles de la
    distribución real observada, en vez de umbrales asumidos. Se usan
    los percentiles [percentile_margin, 100 - percentile_margin] como
    límites, dejando fuera del rango los casos atípicos genuinos.
    """
    area_low = float(np.percentile(area_ratios, percentile_margin))
    area_high = float(np.percentile(area_ratios, 100 - percentile_margin))
    ar_low = float(np.percentile(aspect_ratios, percentile_margin))
    ar_high = float(np.percentile(aspect_ratios, 100 - percentile_margin))

    logger.info(
        "Rangos de validación calibrados (percentil %.1f-%.1f): "
        "area_ratio=[%.4f, %.4f], aspect_ratio=[%.3f, %.3f]",
        percentile_margin, 100 - percentile_margin, area_low, area_high, ar_low, ar_high,
    )
    return (area_low, area_high), (ar_low, ar_high)


def run_second_pass(
    dataset: DatasetInfo,
    detections: dict[str, dict],
    area_ratio_range: tuple[float, float],
    aspect_ratio_range: tuple[float, float],
    margin_ratio: float,
    output_dir: Path,
) -> dict:
    """
    Aplica validación + crop + transformación de boxes a cada imagen,
    usando la detección ya calculada en la Pasada 1 (no se re-detecta).
    Materializa el dataset corregido y produce el reporte final.
    """

    excluded: list[dict] = []
    processed_count = 0
    total_boxes_before = 0
    total_boxes_after = 0
    total_boxes_dropped = 0

    for i, image in enumerate(dataset.images, start=1):
        det = detections[str(image.image_path)]
        total_boxes_before += len(image.boxes)

        if not det["success"]:
            excluded.append({"file": str(image.image_path), "reason": det["failure_reason"]})
            continue

        from balsa_vision.preprocess.panel_cropper import PanelRect  # import local para claridad

        panel_rect = PanelRect(center=tuple(det["center"]), size=tuple(det["size"]), angle=det["angle"])

        # Reconstruir un PanelDetectionResult mínimo para reusar validate_panel_geometry.
        from balsa_vision.preprocess.panel_cropper import PanelDetectionResult

        detection_result = PanelDetectionResult(
            success=True, panel_rect=panel_rect, area_ratio=det["area_ratio"]
        )
        is_valid, reason = validate_panel_geometry(detection_result, area_ratio_range, aspect_ratio_range)

        if not is_valid:
            excluded.append({"file": str(image.image_path), "reason": reason})
            continue

        img = cv2.imread(str(image.image_path))
        crop_result = crop_panel(img, image.boxes, panel_rect, margin_ratio)

        if not crop_result.success:
            excluded.append({"file": str(image.image_path), "reason": crop_result.failure_reason})
            continue

        dest_image_dir = output_dir / "images" / image.split
        dest_label_dir = output_dir / "labels" / image.split
        dest_image_dir.mkdir(parents=True, exist_ok=True)
        dest_label_dir.mkdir(parents=True, exist_ok=True)

        assert crop_result.cropped_image is not None
        cv2.imwrite(str(dest_image_dir / image.image_path.name), crop_result.cropped_image)

        label_lines = [
            f"{b.class_id} {b.x_center:.6f} {b.y_center:.6f} {b.width:.6f} {b.height:.6f}"
            for b in crop_result.transformed_boxes
        ]
        (dest_label_dir / f"{image.image_path.stem}.txt").write_text("\n".join(label_lines), encoding="utf-8")

        processed_count += 1
        total_boxes_after += len(crop_result.transformed_boxes)
        total_boxes_dropped += crop_result.dropped_box_count

        if i % 200 == 0:
            logger.info("Pasada 2 (crop): %d/%d", i, len(dataset.images))

    return {
        "total_images": len(dataset.images),
        "processed_successfully": processed_count,
        "excluded_count": len(excluded),
        "excluded_detail": excluded,
        "total_boxes_before": total_boxes_before,
        "total_boxes_after": total_boxes_after,
        "total_boxes_dropped_at_edges": total_boxes_dropped,
    }


def write_data_yaml(output_dir: Path, class_names: dict[int, str]) -> None:
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in sorted(class_names.items())},
    }
    with (output_dir / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase: detección, enderezado y recorte del panel.")
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--ndjson", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--margin-ratio", type=float, default=0.06, help="Margen de seguridad del crop.")
    parser.add_argument(
        "--percentile-margin", type=float, default=5.0,
        help="Percentil usado en cada extremo para calibrar rangos de validación (default: 5 -> P5-P95).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(root_dir=args.root_dir, ndjson_path=args.ndjson)

    logger.info("=== Pasada 1: detección y calibración de rangos de validación ===")
    detections, area_ratios, aspect_ratios = run_first_pass(dataset)

    detection_failures = sum(1 for d in detections.values() if not d["success"])
    logger.info(
        "Detección completada: %d/%d exitosas, %d fallidas en la fase de detección pura.",
        len(detections) - detection_failures, len(detections), detection_failures,
    )

    area_range, aspect_range = calibrate_validation_ranges(area_ratios, aspect_ratios, args.percentile_margin)

    logger.info("=== Pasada 2: validación + crop + transformación de boxes ===")
    report = run_second_pass(
        dataset, detections, area_range, aspect_range, args.margin_ratio, args.output_dir
    )

    write_data_yaml(args.output_dir, dataset.class_names)

    report["calibrated_area_ratio_range"] = area_range
    report["calibrated_aspect_ratio_range"] = aspect_range
    report["margin_ratio_used"] = args.margin_ratio
    report["percentile_margin_used"] = args.percentile_margin

    report_path = args.output_dir / "panel_crop_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("=" * 70)
    logger.info("REPORTE FINAL DE CROP DE PANEL")
    logger.info("=" * 70)
    logger.info("Total imágenes              : %d", report["total_images"])
    logger.info("Procesadas exitosamente      : %d", report["processed_successfully"])
    logger.info("Excluidas (revisión manual)  : %d", report["excluded_count"])
    logger.info("Boxes antes / después / descartadas en bordes: %d / %d / %d",
                report["total_boxes_before"], report["total_boxes_after"], report["total_boxes_dropped_at_edges"])
    logger.info("Reporte completo guardado en: %s", report_path)
    logger.info("Dataset recortado listo en  : %s", args.output_dir)


if __name__ == "__main__":
    main()