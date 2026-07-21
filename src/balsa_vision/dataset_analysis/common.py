# dataset_analysis/common.py
"""
Módulo compartido para el análisis exploratorio del dataset YOLO.

Centraliza la carga de la estructura del dataset (imágenes, labels,
mapeo de clases) para que los distintos scripts de reporte (distribución
de clases, geometría de bboxes, calidad de imagen, duplicados) no
dupliquen lógica de parseo y permanezcan consistentes entre sí.

Fuente de verdad para el mapeo de clases (ver ADR-005):
    Se prioriza el registro de metadata del .ndjson original sobre
    cualquier archivo generado por el conversor (data.yaml,
    class_names.json), ya que este último no está garantizado por la
    función oficial de Ultralytics y su ausencia no debe romper el
    análisis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box en formato YOLO normalizado."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def area_ratio(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0.0


@dataclass(frozen=True)
class DatasetImage:
    """Representa una imagen del dataset junto con sus anotaciones y split."""

    image_path: Path
    label_path: Path
    split: str
    boxes: list[BoundingBox]

    @property
    def is_negative(self) -> bool:
        return len(self.boxes) == 0

    @property
    def classes_present(self) -> set[int]:
        return {box.class_id for box in self.boxes}


@dataclass(frozen=True)
class DatasetInfo:
    """Estructura completa del dataset cargado, lista para ser consumida por los reportes."""

    root_dir: Path
    class_names: dict[int, str]
    images: list[DatasetImage]

    def images_by_split(self, split: str) -> list[DatasetImage]:
        return [img for img in self.images if img.split == split]


def parse_label_file(label_path: Path) -> list[BoundingBox]:
    """
    Parsea un archivo .txt de labels en formato YOLO. Un archivo vacío o
    inexistente representa una imagen negativa (sin defectos) y devuelve
    una lista vacía, no un error.
    """
    if not label_path.exists():
        return []

    boxes: list[BoundingBox] = []
    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return boxes

    for line_number, line in enumerate(content.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            logger.warning(
                "Línea %d de %s tiene %d campos (se esperaban 5), se omite.",
                line_number, label_path.name, len(parts),
            )
            continue
        class_id, xc, yc, w, h = parts
        boxes.append(
            BoundingBox(
                class_id=int(class_id),
                x_center=float(xc),
                y_center=float(yc),
                width=float(w),
                height=float(h),
            )
        )
    return boxes


def load_class_names_from_ndjson(ndjson_path: Path) -> dict[int, str]:
    """
    Lee el registro de metadata (type == 'dataset') del .ndjson original
    para obtener el mapeo class_id -> nombre. Esta es la fuente de verdad
    autoritativa (ver ADR-005), independiente de lo que haya generado el
    conversor a formato YOLO en disco.
    """
    with ndjson_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") == "dataset" or "class_names" in record:
                raw_names = record["class_names"]
                return {int(k): v for k, v in raw_names.items()}

    raise ValueError(
        f"No se encontró un registro de metadata con 'class_names' en {ndjson_path}. "
        "Verifica que sea el archivo .ndjson original exportado de Ultralytics HUB."
    )


def load_class_names_from_data_yaml(data_yaml_path: Path) -> dict[int, str]:
    """Fallback: lee el mapeo de clases desde un data.yaml estándar, si existe."""
    with data_yaml_path.open("r", encoding="utf-8") as f:
        data_yaml = yaml.safe_load(f)
    names_raw = data_yaml["names"]
    return (
        {int(k): v for k, v in names_raw.items()}
        if isinstance(names_raw, dict)
        else dict(enumerate(names_raw))
    )


def load_dataset(
    root_dir: Path,
    ndjson_path: Path | None = None,
    data_yaml_path: Path | None = None,
) -> DatasetInfo:
    """
    Carga la estructura completa del dataset.

    El mapeo de clases se resuelve en este orden de prioridad explícito
    (ver ADR-005), sin asumir silenciosamente ningún fallback:
        1. ndjson_path, si se provee (fuente de verdad recomendada).
        2. data_yaml_path, si se provee y existe.
        3. Error explícito si ninguna fuente fue provista — nunca se
           inventan nombres de clase genéricos.
    """
    if ndjson_path is not None:
        class_names = load_class_names_from_ndjson(ndjson_path)
    elif data_yaml_path is not None and data_yaml_path.exists():
        class_names = load_class_names_from_data_yaml(data_yaml_path)
    else:
        raise ValueError(
            "No se proveyó una fuente válida para el mapeo de clases. "
            "Pasa --ndjson apuntando al archivo .ndjson original, o "
            "--data-yaml si ya cuentas con un data.yaml estándar."
        )

    images: list[DatasetImage] = []
    for split in ("train", "val", "test"):
        images_dir = root_dir / "images" / split
        labels_dir = root_dir / "labels" / split
        if not images_dir.exists():
            continue

        for image_path in sorted(images_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            boxes = parse_label_file(label_path)
            images.append(
                DatasetImage(
                    image_path=image_path,
                    label_path=label_path,
                    split=split,
                    boxes=boxes,
                )
            )

    if not images:
        raise FileNotFoundError(
            f"No se encontraron imágenes bajo {root_dir}/images/<split>/. "
            "Verifica la estructura real del directorio con un listado manual "
            "antes de continuar (los nombres de split podrían no ser "
            "'train'/'val' exactamente)."
        )

    logger.info(
        "Dataset cargado: %d imágenes totales, %d clases (%s).",
        len(images), len(class_names), class_names,
    )
    return DatasetInfo(root_dir=root_dir, class_names=class_names, images=images)