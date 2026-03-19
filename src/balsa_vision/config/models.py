from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class AppConfig:
    project_name: str
    results_dir: str
    save_raw: bool
    save_annotated: bool
    log_format: str  # csv|jsonl|both


@dataclass(frozen=True)
class ModelConfig:
    model_path: str
    input_size: int
    providers: List[str]


@dataclass(frozen=True)
class ThresholdsConfig:
    conf_thres: float
    iou_thres: float


@dataclass(frozen=True)
class ClassSpec:
    name: str
    critical: bool


@dataclass(frozen=True)
class ClassesConfig:
    classes: List[ClassSpec]


@dataclass(frozen=True)
class Settings:
    app: AppConfig
    model: ModelConfig
    thresholds: ThresholdsConfig
    classes: ClassesConfig