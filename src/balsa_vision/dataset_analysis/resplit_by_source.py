# dataset_analysis/resplit_by_source.py
"""
Genera un nuevo split train/val agrupado por fuente Roboflow
(source_key extraído del patrón "<original>.rf.<hash>.jpg"), garantizando
que todas las variantes augmentadas de un mismo panel físico permanezcan
en el mismo split. Corrige el leakage confirmado en ADR-002.

Estrategia:
    - Se usa StratifiedGroupKFold (scikit-learn >= 1.1) para mantener
      grupos juntos y, en la medida de lo posible, preservar la
      proporción de clases entre splits.
    - Las imágenes sin agrupación real (direct_capture, single-image
      groups) participan igualmente del proceso: cada una es su propio
      grupo de tamaño 1, sin restricción adicional.
    - La etiqueta de estratificación por grupo se define como la
      combinación de clases presentes en las anotaciones del grupo:
      "negative", "nudillo_only", "sheck_only", "both". Esto evita que
      el re-split concentre por azar todos los negativos o toda una
      clase en un solo split.
    - El script NO mueve ni borra el dataset original: genera una nueva
      carpeta con estructura YOLO estándar (images/, labels/, data.yaml)
      reflejando el split corregido, copiando (no symlink, para
      portabilidad en Windows) los archivos correspondientes.

Requiere: pip install scikit-learn>=1.1

Uso:
    python -m dataset_analysis.resplit_by_source \
        --root-dir /ruta/al/dataset/b8768037 \
        --ndjson /ruta/al/archivo.ndjson \
        --output-dir /ruta/al/dataset/dataset_yolo_resplit \
        --val-ratio 0.2 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from balsa_vision.dataset_analysis.common import DatasetImage, DatasetInfo, load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

ROBOFLOW_PATTERN = re.compile(r"^(.+)\.rf\.[0-9a-fA-F]+\.(jpg|jpeg|png)$", re.IGNORECASE)


def extract_source_key(filename: str) -> str:
    """
    Misma lógica que roboflow_source_leakage.py: agrupa variantes
    Roboflow por nombre original; cualquier otro patrón se trata como
    fuente única (grupo de tamaño 1).
    """
    match = ROBOFLOW_PATTERN.match(filename)
    return match.group(1) if match else filename


def build_group_labels(dataset: DatasetInfo) -> tuple[list[str], list[str], list[str]]:
    """
    Construye tres listas paralelas (una entrada por imagen) requeridas
    por StratifiedGroupKFold:
        - image_keys: identificador único de cada imagen (ruta como str).
        - groups: source_key de cada imagen (para mantenerlas juntas).
        - strat_labels: etiqueta de estratificación por imagen, derivada
          de las clases presentes en ESA imagen (no del grupo completo;
          StratifiedGroupKFold agrega internamente por grupo).
    """
    image_keys: list[str] = []
    groups: list[str] = []
    strat_labels: list[str] = []

    for image in dataset.images:
        image_keys.append(str(image.image_path))
        groups.append(extract_source_key(image.image_path.name))

        classes_present = image.classes_present
        if not classes_present:
            label = "negative"
        elif classes_present == {0}:
            label = "nudillo_only"
        elif classes_present == {1}:
            label = "sheck_only"
        else:
            label = "both"
        strat_labels.append(label)

    return image_keys, groups, strat_labels


def compute_resplit(
    dataset: DatasetInfo, val_ratio: float, seed: int
) -> dict[str, str]:
    """
    Ejecuta StratifiedGroupKFold y devuelve un mapeo {image_path_str: new_split}.

    n_splits se deriva de val_ratio (ej. val_ratio=0.2 -> n_splits=5,
    se toma 1 fold como val). No es una proporción exacta garantizada
    (por la restricción de grupos), pero se reporta la proporción real
    resultante para verificación.
    """
    image_keys, groups, strat_labels = build_group_labels(dataset)
    n_splits = round(1 / val_ratio)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    X = np.zeros(len(image_keys))
    y = np.array(strat_labels)
    g = np.array(groups)

    train_idx, val_idx = next(splitter.split(X, y, groups=g))

    new_split: dict[str, str] = {}
    for idx in train_idx:
        new_split[image_keys[idx]] = "train"
    for idx in val_idx:
        new_split[image_keys[idx]] = "val"

    return new_split


def verify_no_leakage(dataset: DatasetInfo, new_split: dict[str, str]) -> int:
    """
    Verificación de seguridad post-split: confirma que ningún source_key
    quede repartido entre train y val en el nuevo split. Devuelve la
    cantidad de grupos con leakage residual (debe ser 0).
    """
    group_splits: dict[str, set[str]] = defaultdict(set)
    for image in dataset.images:
        key = extract_source_key(image.image_path.name)
        group_splits[key].add(new_split[str(image.image_path)])

    leaking = sum(1 for splits in group_splits.values() if len(splits) > 1)
    return leaking


def report_class_balance(dataset: DatasetInfo, new_split: dict[str, str]) -> dict:
    """Genera un resumen de la nueva distribución de clases por split, para verificación."""
    counts: dict[str, dict[str, int]] = {
        "train": defaultdict(int),
        "val": defaultdict(int),
    }
    for image in dataset.images:
        split = new_split[str(image.image_path)]
        if image.is_negative:
            counts[split]["negative"] += 1
        else:
            for class_id in image.classes_present:
                class_name = dataset.class_names[class_id]
                counts[split][class_name] += 1

    return {split: dict(class_counts) for split, class_counts in counts.items()}


def materialize_new_dataset(
    dataset: DatasetInfo, new_split: dict[str, str], output_dir: Path
) -> None:
    """
    Copia cada imagen y su label al nuevo split corregido, generando la
    estructura estándar YOLO (images/<split>/, labels/<split>/).
    """
    for image in dataset.images:
        new_split_name = new_split[str(image.image_path)]

        dest_image_dir = output_dir / "images" / new_split_name
        dest_label_dir = output_dir / "labels" / new_split_name
        dest_image_dir.mkdir(parents=True, exist_ok=True)
        dest_label_dir.mkdir(parents=True, exist_ok=True)

        dest_image_path = dest_image_dir / image.image_path.name
        if not dest_image_path.exists():
            shutil.copy2(image.image_path, dest_image_path)

        dest_label_path = dest_label_dir / f"{image.image_path.stem}.txt"
        if image.label_path.exists():
            shutil.copy2(image.label_path, dest_label_path)
        else:
            # Negativo intencional: label vacío explícito (consistente con ADR-001).
            dest_label_path.write_text("", encoding="utf-8")


def write_data_yaml(output_dir: Path, class_names: dict[int, str]) -> None:
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in sorted(class_names.items())},
    }
    yaml_path = output_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)
    logger.info("data.yaml generado en %s", yaml_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-split agrupado por fuente Roboflow, corrigiendo el "
        "leakage confirmado en ADR-002."
    )
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--ndjson", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_dataset(root_dir=args.root_dir, ndjson_path=args.ndjson)

    logger.info("Calculando re-split agrupado y estratificado (seed=%d)...", args.seed)
    new_split = compute_resplit(dataset, args.val_ratio, args.seed)

    leaking_groups = verify_no_leakage(dataset, new_split)
    if leaking_groups > 0:
        raise RuntimeError(
            f"Verificación de seguridad falló: {leaking_groups} grupos aún "
            "tienen leakage tras el re-split. No se materializa el dataset."
        )
    logger.info("✅ Verificación de seguridad: 0 grupos con leakage en el nuevo split.")

    balance_report = report_class_balance(dataset, new_split)
    total_train = sum(balance_report["train"].values())
    total_val = sum(balance_report["val"].values())
    logger.info("=" * 70)
    logger.info("BALANCE DE CLASES TRAS EL RE-SPLIT")
    logger.info("=" * 70)
    logger.info("Train: %s (total anotaciones+negativos: %d)", balance_report["train"], total_train)
    logger.info("Val  : %s (total anotaciones+negativos: %d)", balance_report["val"], total_val)

    n_train_images = sum(1 for s in new_split.values() if s == "train")
    n_val_images = sum(1 for s in new_split.values() if s == "val")
    logger.info(
        "Proporción real de imágenes: train=%d (%.1f%%), val=%d (%.1f%%)",
        n_train_images, 100 * n_train_images / len(new_split),
        n_val_images, 100 * n_val_images / len(new_split),
    )

    logger.info("Materializando nuevo dataset en: %s", args.output_dir)
    materialize_new_dataset(dataset, new_split, args.output_dir)
    write_data_yaml(args.output_dir, dataset.class_names)

    summary_path = args.output_dir / "resplit_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "val_ratio_requested": args.val_ratio,
                "val_ratio_actual": n_val_images / len(new_split),
                "seed": args.seed,
                "leaking_groups_after_resplit": leaking_groups,
                "class_balance": balance_report,
                "n_train_images": n_train_images,
                "n_val_images": n_val_images,
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("Resumen guardado en: %s", summary_path)
    logger.info("✅ Re-split completo. Dataset corregido listo en: %s", args.output_dir)


if __name__ == "__main__":
    main()