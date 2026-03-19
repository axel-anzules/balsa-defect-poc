from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

from .models import (
    AppConfig,
    ClassSpec,
    ClassesConfig,
    ModelConfig,
    Settings,
    ThresholdsConfig,
)


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in {path}: expected mapping at root")
    return data


def load_settings(config_dir: str) -> Settings:
    cfg_dir = Path(config_dir)

    app_d = _read_yaml(cfg_dir / "app.yaml")
    model_d = _read_yaml(cfg_dir / "model.yaml")
    thr_d = _read_yaml(cfg_dir / "thresholds.yaml")
    cls_d = _read_yaml(cfg_dir / "classes.yaml")

    app = AppConfig(
        project_name=str(app_d.get("project_name", "balsa-defect-poc")),
        results_dir=str(app_d.get("results_dir", "results")),
        save_raw=bool(app_d.get("save_raw", True)),
        save_annotated=bool(app_d.get("save_annotated", True)),
        log_format=str(app_d.get("log_format", "jsonl")),
    )

    model = ModelConfig(
        model_path=str(model_d["model_path"]),
        input_size=int(model_d.get("input_size", 640)),
        providers=list(model_d.get("providers", ["CPUExecutionProvider"])),
    )

    thresholds = ThresholdsConfig(
        conf_thres=float(thr_d.get("conf_thres", 0.25)),
        iou_thres=float(thr_d.get("iou_thres", 0.45)),
    )

    classes_raw = cls_d.get("classes", [])
    classes: List[ClassSpec] = [
        ClassSpec(name=str(c["name"]), critical=bool(c.get("critical", False)))
        for c in classes_raw
    ]
    classes_cfg = ClassesConfig(classes=classes)

    return Settings(app=app, model=model, thresholds=thresholds, classes=classes_cfg)