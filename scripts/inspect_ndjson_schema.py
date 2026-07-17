"""
inspect_ndjson_schema.py

Utilidad de reconocimiento de esquema para exports .ndjson de Ultralytics HUB.
No asume estructura: reporta las claves presentes, tipos de dato y una muestra
cruda de los primeros registros para poder diseñar el conversor a formato YOLO
sobre datos reales.

Uso:
    python inspect_ndjson_schema.py --input /ruta/al/archivo.ndjson

Sin dependencias externas (solo librería estándar) por diseño: este es un
script de diagnóstico de una sola ejecución, no forma parte del pipeline
final, por lo que no amerita añadir dependencias al entorno todavía.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def collect_key_signature(record: dict[str, Any], prefix: str = "") -> set[str]:
    """
    Devuelve el conjunto de rutas de claves (dot-notation) presentes en un
    registro JSON, recorriendo diccionarios anidados de forma recursiva.

    Esto permite detectar si distintos registros del ndjson tienen esquemas
    ligeramente distintos (por ejemplo, imágenes sin anotaciones que omiten
    el campo 'annotations').
    """
    keys: set[str] = set()
    for key, value in record.items():
        full_key = f"{prefix}{key}"
        keys.add(full_key)
        if isinstance(value, dict):
            keys |= collect_key_signature(value, prefix=f"{full_key}.")
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            keys |= collect_key_signature(value[0], prefix=f"{full_key}[].")
    return keys


def inspect_ndjson(input_path: Path, sample_size: int = 3) -> None:
    """
    Recorre el archivo .ndjson línea por línea (streaming, apto para archivos
    grandes) y reporta:
      - Cantidad total de registros (líneas válidas).
      - Firmas de esquema únicas encontradas (para detectar inconsistencias).
      - Frecuencia de cada firma de esquema.
      - Muestra cruda (pretty-printed) de los primeros N registros.
      - Conteo de líneas que fallaron el parseo JSON, si las hay.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

    schema_signatures: Counter[frozenset[str]] = Counter()
    total_records = 0
    parse_errors = 0
    sample_records: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors += 1
                logger.warning("Línea %d no es JSON válido: %s", line_number, exc)
                continue

            total_records += 1
            signature = frozenset(collect_key_signature(record))
            schema_signatures[signature] += 1

            if len(sample_records) < sample_size:
                sample_records.append(record)

    logger.info("=" * 70)
    logger.info("REPORTE DE ESQUEMA — %s", input_path.name)
    logger.info("=" * 70)
    logger.info("Registros totales válidos : %d", total_records)
    logger.info("Líneas con error de parseo : %d", parse_errors)
    logger.info("Firmas de esquema distintas: %d", len(schema_signatures))
    logger.info("-" * 70)

    for idx, (signature, count) in enumerate(
        sorted(schema_signatures.items(), key=lambda x: -x[1]), start=1
    ):
        logger.info("Firma #%d — aparece en %d registros:", idx, count)
        for key in sorted(signature):
            logger.info("    - %s", key)
        logger.info("-" * 70)

    logger.info("MUESTRA CRUDA DE REGISTROS (primeros %d):", len(sample_records))
    for idx, record in enumerate(sample_records, start=1):
        logger.info("--- Registro de muestra %d ---", idx)
        logger.info(json.dumps(record, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspecciona el esquema de un archivo .ndjson exportado "
        "de Ultralytics HUB antes de diseñar el conversor a formato YOLO."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Ruta al archivo .ndjson descargado.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="Cantidad de registros crudos a imprimir como muestra (default: 3).",
    )
    args = parser.parse_args()

    inspect_ndjson(input_path=args.input, sample_size=args.sample_size)


if __name__ == "__main__":
    main()