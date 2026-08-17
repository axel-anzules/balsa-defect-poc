from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import yaml
from ultralytics import YOLO

import torch

from balsa_vision.training.custom_albumentations import APPROVED_TRANSFORMS
from balsa_vision.training.experiment_tracking import log_experiment_result

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

FIXED_TRAIN_ARGS: dict = {
    "optimizer": "auto",
    "amp": True,
    "cache": "ram",
    "device": "0",
    "batch": -1,
    "close_mosaic": 15,
}

def load_hyp(hyp_path: Path) -> dict:
    with hyp_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def extract_metrics(train_results, class_names: dict[int, str]) -> dict:
    box = train_results.box
    per_class_map = {}
    for idx, name in class_names.items():
        try:
            per_class_map[f"map50_95_{name}"] = float(box.maps[idx])
        except (IndexError, AttributeError):
            per_class_map[f"map50_95_{name}"] = None

    return {
        "precision": float(box.mp),
        "recal": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        **per_class_map,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Runner de corrida expermiental.")
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, help="ej. yolo11n")
    parser.add_argument("--data-root", type=Path, required=True, help="Carpeta con dara.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--rect", action="store_true", help="Habilita entrenamiento rectangular")
    parser.add_argument("--hyp", type=Path, required=True)
    parser.add_argument("--clahe-label", type=str, default="con_clahe",
                        help="Etiqueta descriptiva del dataset usado (con/sin CLAHE) para el log.")
    parser.add_argument("--results-csv", type=Path,
                        default=Path("reports/experiments/experiment_results.csv"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--box", type=float, default=7.5)
    parser.add_argument("--dfl", type=float, default=1.5)
    parser.add_argument("--cos-lr", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    args = parser.parse_args()

    hyp = load_hyp(args.hyp)
    data_yaml = args.data_root / "data.yaml"

    with data_yaml.open("r", encoding="utf-8") as f:
        class_names = {int(k): v for k, v in yaml.safe_load(f)["names"].items()}

    if args.rect:
        logger.warning(
            "rect=True activo: Mosaic y MixUp quedan desactivados automaticamente por Ultralytics"
        )
    model = YOLO(args.model)

    logger.info("=== Iniciando corrida %s ===", args.experiment_id)
    start_time = time.monotonic()

    train_results = model.train(
        torch.cuda.empty_cache(),
        data=str(data_yaml),
        imgsz=args.imgsz,
        rect=args.rect,
        name=args.experiment_id,
        workers=args.workers,
        augmentations=APPROVED_TRANSFORMS,
        **hyp,
        **FIXED_TRAIN_ARGS,
    )

    elapsed_minutos = (time.monotonic() - start_time) / 60.0

    metrics = extract_metrics(train_results, class_names)
    metrics["train_time_minutes"] = round(elapsed_minutos, 1)

    config = {
        "model": args.model,
        "imgsz": args.imgsz,
        "rect": args.rect,
        "clahe": args.clahe_label,
        "augmentation": args.hyp.stem,
    }

    log_experiment_result(
        results_csv_path=args.results_csv,
        experiment_id=args.experiment_id,
        config=config,
        metrics=metrics,
        run_dir=str(train_results.save_dir),
    )

    logger.info("=== Corrida %s completa ===", args.experiment_id)
    logger.info("mAP50-95 global: %.4f | Tiempo: %.1f min", metrics["map50_95"], elapsed_minutos)
    logger.info("Resultados completos en: %s", train_results.save_dir)

if __name__ == "__main__":
    main()