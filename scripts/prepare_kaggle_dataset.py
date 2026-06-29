from __future__ import annotations

from pathlib import Path
import random
import shutil

RANDOM_SEED = 42
VAL_RATIO = 0.2

SRC_IMAGES = Path("external_data/kaggle_wood_defects/Images - 1")
SRC_LABELS = Path("external_data/kaggle_wood_defects/Bounding Boxes - YOLO Format - 1")

DST_ROOT = Path("dataset_kaggle_test")
DST_IMAGES_TRAIN = DST_ROOT / "images" / "train"
DST_IMAGES_VAL = DST_ROOT / "images" / "val"
DST_LABELS_TRAIN = DST_ROOT / "labels" / "train"
DST_LABELS_VAL = DST_ROOT / "labels" / "val"


def main() -> None:
    for p in [
        DST_IMAGES_TRAIN,
        DST_IMAGES_VAL,
        DST_LABELS_TRAIN,
        DST_LABELS_VAL,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(SRC_IMAGES.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No se encontraron imágenes en: {SRC_IMAGES}")

    valid_pairs = []
    missing_labels = []

    for img_path in image_paths:
        label_path = SRC_LABELS / f"{img_path.stem}.txt"
        if label_path.exists():
            valid_pairs.append((img_path, label_path))
        else:
            missing_labels.append(img_path.name)

    if not valid_pairs:
        raise RuntimeError("No se encontraron pares imagen-label válidos.")

    print(f"Total imágenes con label: {len(valid_pairs)}")
    print(f"Total imágenes sin label: {len(missing_labels)}")

    random.seed(RANDOM_SEED)
    random.shuffle(valid_pairs)

    split_idx = int(len(valid_pairs) * (1 - VAL_RATIO))
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]

    for img_path, label_path in train_pairs:
        shutil.copy2(img_path, DST_IMAGES_TRAIN / img_path.name)
        shutil.copy2(label_path, DST_LABELS_TRAIN / label_path.name)

    for img_path, label_path in val_pairs:
        shutil.copy2(img_path, DST_IMAGES_VAL / img_path.name)
        shutil.copy2(label_path, DST_LABELS_VAL / label_path.name)

    print(f"Train: {len(train_pairs)}")
    print(f"Val: {len(val_pairs)}")
    print("Dataset Kaggle preparado correctamente.")


if __name__ == "__main__":
    main()