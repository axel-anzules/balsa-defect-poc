"""
resume_dataset_download.py

Detecta huecos en un dataset ya convertido con
ultralytics.data.converter.convert_ndjson_to_yolo (imágenes que no se
descargaron correctamente) y los reintenta de forma dirigida, sin volver
a procesar el dataset completo.

También detecta labels "huérfanos" (archivo .txt presente sin imagen
correspondiente), que son un riesgo silencioso para el entrenamiento.

Diseño:
    - Concurrencia baja por defecto (4 workers) y timeout amplio, para
      evitar volver a disparar el rate limiting observado en el CDN.
    - Backoff exponencial con jitter entre reintentos.
    - Idempotente: si la imagen ya existe y tiene tamaño > 0, se omite.
    - Verifica integridad básica de cada imagen descargada (que OpenCV
      pueda decodificarla), no solo que el archivo exista, ya que una
      descarga cortada puede dejar un archivo presente pero corrupto.

Uso:
    python resume_dataset_download.py \
        --input dataset.ndjson \
        --dataset-dir "C:\\...\\dataset\\defects-on-wood-balsa-panels-version-with-preprocess-b8768037" \
        --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT_SECONDS = 30
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 2.0


@dataclass
class IntegrityReport:
    total_expected: int = 0
    already_present_ok: int = 0
    corrupted_and_redownloaded: int = 0
    newly_downloaded: int = 0
    still_missing: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)
    final_present_count: int = 0


def is_metadata_record(record: dict[str, Any]) -> bool:
    return record.get("type") == "dataset" or "class_names" in record


def load_image_records(input_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if is_metadata_record(record):
                continue
            records.append(record)
    return records


def is_image_valid(path: Path) -> bool:
    """
    Verifica que el archivo exista, tenga tamaño no nulo, y que OpenCV
    pueda decodificarlo. Una descarga cortada por 'Server disconnected'
    puede dejar un archivo presente pero con bytes incompletos/corruptos.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    img = cv2.imread(str(path))
    return img is not None


def download_with_backoff(url: str, dest_path: Path) -> tuple[bool, str | None]:
    """Descarga con reintentos y backoff exponencial + jitter."""
    last_error: str | None = None
    session = requests.Session()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, timeout=DOWNLOAD_TIMEOUT_SECONDS)
            response.raise_for_status()
            dest_path.write_bytes(response.content)
            if is_image_valid(dest_path):
                return True, None
            last_error = "descarga completada pero imagen no decodificable"
        except requests.RequestException as exc:
            last_error = str(exc)

        if attempt < MAX_RETRIES:
            sleep_time = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 1)
            logger.warning(
                "Intento %d/%d falló para %s (%s). Reintentando en %.1fs...",
                attempt, MAX_RETRIES, dest_path.name, last_error, sleep_time,
            )
            time.sleep(sleep_time)

    return False, last_error


def resolve_image_path(dataset_dir: Path, split: str, filename: str) -> Path:
    return dataset_dir / "images" / split / filename


def resolve_label_path(dataset_dir: Path, split: str, filename: str) -> Path:
    return dataset_dir / "labels" / split / f"{Path(filename).stem}.txt"


def process_missing_record(
    record: dict[str, Any], dataset_dir: Path
) -> tuple[str, bool, str | None]:
    """Reintenta la descarga de un único registro faltante o corrupto."""
    split = record.get("split", "train")
    filename = record["file"]
    image_path = resolve_image_path(dataset_dir, split, filename)
    success, error = download_with_backoff(record["url"], image_path)
    return filename, success, error


def run_integrity_check_and_resume(
    input_path: Path, dataset_dir: Path, workers: int
) -> IntegrityReport:
    records = load_image_records(input_path)
    report = IntegrityReport(total_expected=len(records))

    to_retry: list[dict[str, Any]] = []
    expected_filenames: set[str] = set()

    for record in records:
        split = record.get("split", "train")
        filename = record["file"]
        expected_filenames.add(filename)
        image_path = resolve_image_path(dataset_dir, split, filename)

        if is_image_valid(image_path):
            report.already_present_ok += 1
        else:
            if image_path.exists():
                report.corrupted_and_redownloaded += 1
                logger.warning("Imagen corrupta detectada, se reintentará: %s", filename)
            to_retry.append(record)

    logger.info(
        "Estado inicial: %d OK | %d a reintentar (faltantes o corruptas) de %d totales",
        report.already_present_ok, len(to_retry), report.total_expected,
    )

    if to_retry:
        logger.info("Reintentando descarga con %d workers (concurrencia reducida)...", workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_missing_record, record, dataset_dir): record["file"]
                for record in to_retry
            }
            for i, future in enumerate(as_completed(futures), start=1):
                filename, success, error = future.result()
                if success:
                    report.newly_downloaded += 1
                else:
                    report.still_missing.append(filename)
                    logger.error("Definitivamente no se pudo descargar %s: %s", filename, error)
                if i % 20 == 0:
                    logger.info("Progreso de reintento: %d/%d", i, len(to_retry))

    # Detección de labels huérfanos: archivo .txt presente cuya imagen
    # correspondiente sigue ausente tras el reintento.
    for split_dir_name in ("train", "val"):
        labels_dir = dataset_dir / "labels" / split_dir_name
        images_dir = dataset_dir / "images" / split_dir_name
        if not labels_dir.exists():
            continue
        for label_file in labels_dir.glob("*.txt"):
            stem = label_file.stem
            matching_images = list(images_dir.glob(f"{stem}.*"))
            if not any(is_image_valid(p) for p in matching_images):
                report.orphan_labels.append(str(label_file))

    report.final_present_count = report.already_present_ok + report.newly_downloaded

    return report


def print_report(report: IntegrityReport) -> None:
    logger.info("=" * 70)
    logger.info("REPORTE DE INTEGRIDAD Y RECUPERACIÓN DE DESCARGA")
    logger.info("=" * 70)
    logger.info("Total esperado según ndjson      : %d", report.total_expected)
    logger.info("Ya presentes y válidas (inicial)  : %d", report.already_present_ok)
    logger.info("Corruptas detectadas y reintentadas: %d", report.corrupted_and_redownloaded)
    logger.info("Recuperadas exitosamente en retry : %d", report.newly_downloaded)
    logger.info("Presentes y válidas (final)       : %d / %d",
                report.final_present_count, report.total_expected)
    logger.info("Aún faltantes tras reintentos      : %d", len(report.still_missing))
    if report.still_missing:
        logger.warning("Archivos que siguen sin descargarse:")
        for fname in report.still_missing:
            logger.warning("    - %s", fname)
    logger.info("Labels huérfanos (sin imagen válida): %d", len(report.orphan_labels))
    if report.orphan_labels:
        logger.warning("Se recomienda NO entrenar hasta resolver estos labels huérfanos:")
        for label_path in report.orphan_labels:
            logger.warning("    - %s", label_path)
    logger.info("=" * 70)

    if not report.still_missing and not report.orphan_labels:
        logger.info("✅ Dataset íntegro: %d/%d imágenes con label consistente. "
                    "Listo para pasar a Fase 1 (análisis exploratorio).",
                    report.final_present_count, report.total_expected)
    else:
        logger.warning("⚠️  Dataset AÚN NO está íntegro. Revisar antes de continuar.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detecta y repara huecos en un dataset YOLO convertido "
        "desde ndjson, sin re-procesar todo el dataset."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    report = run_integrity_check_and_resume(args.input, args.dataset_dir, args.workers)
    print_report(report)


if __name__ == "__main__":
    main()