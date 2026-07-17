"""
convert_ndjson_to_yolo.py

Convierte el export .ndjson de Ultralytics HUB a la estructura estándar
YOLO (images/<split>/, labels/<split>/, data.yaml), descargando las
imágenes desde las URLs firmadas del CDN.

Notas de diseño:
    - Las URLs del CDN incluyen firma con expiración (query param "Expires").
      Este script debe ejecutarse pronto tras la descarga del .ndjson.
    - Cada box se valida (rango [0,1], class_id conocido) antes de
      escribirse; boxes inválidas se registran en el reporte y se excluyen,
      nunca se "corrigen" silenciosamente.
    - Imágenes sin anotaciones generan un archivo .txt vacío explícito
      (convención estándar de Ultralytics para negativos/background),
      en vez de omitir el archivo, para que quede trazable qué imágenes
      fueron procesadas como negativo intencional vs. no procesadas.
    - Descarga concurrente (I/O-bound) con reintentos, idempotente:
      si el archivo ya existe localmente, se omite (permite reanudar).

Uso:
    python convert_ndjson_to_yolo.py \
        --input dataset.ndjson \
        --output-dir ./dataset_yolo \
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT_SECONDS = 20
MAX_RETRIES = 3


@dataclass
class ConversionStats:
    """Acumulador de resultados del proceso de conversión, para el reporte final."""

    total_images: int = 0
    downloaded_ok: int = 0
    download_failed: list[str] = field(default_factory=list)
    boxes_written: int = 0
    boxes_invalid: list[dict[str, Any]] = field(default_factory=list)
    images_per_split: dict[str, int] = field(default_factory=dict)


def load_metadata_and_records(
    input_path: Path,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """
    Carga el registro de metadata (class_names) y todos los registros de
    imagen del .ndjson.
    """
    class_names: dict[str, str] = {}
    records: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") == "dataset" or "class_names" in record:
                class_names = record["class_names"]
                continue
            records.append(record)

    if not class_names:
        raise ValueError(
            "No se encontró el registro de metadata con 'class_names' en el ndjson. "
            "No se puede generar data.yaml sin conocer el mapeo de clases."
        )
    return class_names, records


def validate_box(
    box: list[float], num_classes: int, filename: str
) -> tuple[bool, str | None]:
    """
    Valida una bounding box en formato YOLO [class_id, xc, yc, w, h].
    Devuelve (es_valida, motivo_error_si_invalida).
    """
    if len(box) != 5:
        return False, f"longitud de box inesperada ({len(box)})"

    class_id, xc, yc, w, h = box
    if not float(class_id).is_integer() or not (0 <= int(class_id) < num_classes):
        return False, f"class_id fuera de rango: {class_id}"
    for name, value in (("xc", xc), ("yc", yc), ("w", w), ("h", h)):
        if not (0.0 <= value <= 1.0):
            return False, f"{name} fuera de rango [0,1]: {value}"
    if w <= 0 or h <= 0:
        return False, f"box con área nula/negativa (w={w}, h={h})"
    return True, None


def write_label_file(
    record: dict[str, Any],
    labels_dir: Path,
    num_classes: int,
    stats: ConversionStats,
) -> None:
    """
    Escribe el archivo .txt de etiquetas YOLO para una imagen. Si no hay
    anotaciones, escribe un archivo vacío explícito (negativo intencional).
    """
    filename = record["file"]
    label_path = labels_dir / f"{Path(filename).stem}.txt"
    boxes = record.get("annotations", {}).get("boxes", [])

    lines: list[str] = []
    for box in boxes:
        is_valid, reason = validate_box(box, num_classes, filename)
        if not is_valid:
            logger.warning("Box inválida en %s: %s", filename, reason)
            stats.boxes_invalid.append({"file": filename, "box": box, "reason": reason})
            continue
        class_id, xc, yc, w, h = box
        lines.append(f"{int(class_id)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        stats.boxes_written += 1

    label_path.write_text("\n".join(lines), encoding="utf-8")


def download_image(url: str, dest_path: Path) -> tuple[bool, str | None]:
    """
    Descarga una imagen con reintentos. Si el archivo ya existe (ejecución
    reanudada), no vuelve a descargarlo.
    """
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True, None

    last_error: str | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=DOWNLOAD_TIMEOUT_SECONDS)
            response.raise_for_status()
            dest_path.write_bytes(response.content)
            return True, None
        except requests.RequestException as exc:
            last_error = str(exc)
            logger.warning(
                "Intento %d/%d falló para %s: %s", attempt, MAX_RETRIES, dest_path.name, exc
            )
    return False, last_error


def process_record(
    record: dict[str, Any],
    output_dir: Path,
    num_classes: int,
    stats: ConversionStats,
) -> None:
    """Procesa un único registro: descarga imagen + escribe label."""
    split = record.get("split", "train")
    filename = record["file"]

    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_path = images_dir / filename
    success, error = download_image(record["url"], image_path)

    if not success:
        stats.download_failed.append(filename)
        logger.error("Descarga definitivamente fallida: %s (%s)", filename, error)
        return

    stats.downloaded_ok += 1
    write_label_file(record, labels_dir, num_classes, stats)


def write_data_yaml(output_dir: Path, class_names: dict[str, str]) -> None:
    """Genera data.yaml en formato estándar Ultralytics."""
    sorted_names = [class_names[str(i)] for i in range(len(class_names))]
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(sorted_names)},
    }
    yaml_path = output_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)
    logger.info("data.yaml generado en %s", yaml_path)


def convert(input_path: Path, output_dir: Path, workers: int) -> ConversionStats:
    class_names, records = load_metadata_and_records(input_path)
    num_classes = len(class_names)
    stats = ConversionStats(total_images=len(records))

    logger.info("Clases detectadas: %s", class_names)
    logger.info("Total de imágenes a procesar: %d", stats.total_images)

    output_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_record, record, output_dir, num_classes, stats): record[
                "file"
            ]
            for record in records
        }
        for i, future in enumerate(as_completed(futures), start=1):
            filename = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 — se registra y se continúa
                logger.error("Error inesperado procesando %s: %s", filename, exc)
                stats.download_failed.append(filename)
            if i % 100 == 0:
                logger.info("Progreso: %d/%d", i, stats.total_images)

    for split in ("train", "val"):
        split_dir = output_dir / "images" / split
        stats.images_per_split[split] = (
            len(list(split_dir.glob("*"))) if split_dir.exists() else 0
        )

    write_data_yaml(output_dir, class_names)
    return stats


def write_report(stats: ConversionStats, output_dir: Path) -> None:
    report = {
        "total_images": stats.total_images,
        "downloaded_ok": stats.downloaded_ok,
        "download_failed_count": len(stats.download_failed),
        "download_failed_files": stats.download_failed,
        "boxes_written": stats.boxes_written,
        "boxes_invalid_count": len(stats.boxes_invalid),
        "boxes_invalid_detail": stats.boxes_invalid,
        "images_per_split": stats.images_per_split,
    }
    report_path = output_dir / "conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Reporte de conversión escrito en %s", report_path)
    logger.info("=" * 70)
    logger.info("RESUMEN: %d/%d descargadas OK | %d fallidas | %d boxes inválidas",
                 stats.downloaded_ok, stats.total_images,
                 len(stats.download_failed), len(stats.boxes_invalid))
    logger.info("Imágenes por split: %s", stats.images_per_split)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convierte un export .ndjson de Ultralytics HUB a "
        "estructura estándar YOLO (images/labels/data.yaml)."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    stats = convert(args.input, args.output_dir, args.workers)
    write_report(stats, args.output_dir)


if __name__ == "__main__":
    main()