from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import logging

import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class OnnxYoloConfig:
    model_path: str
    providers: List[str]

class OnnxYoloRunner:
    """Carga de modelo ONNX y ejecuta el inferencia. No hace postproceso"""

    def __init__(self, cfg: OnnxYoloConfig) -> None:
        self.cfg = cfg
        so = ort.SessionOptions()
        # Ajustes conservadores para estabilidad; luego se optimiza para Jetson.
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(cfg.model_path, sess_options=so, providers=cfg.providers)
        self.output_names = [o.name for o in self.session.get_outputs()]
    
    def input_name(self) -> str:
        """Get the input tensor name"""
        return str(self.session.get_inputs()[0].name)

    def warmup(self, input_tensor: np.ndarray) -> None:
        """1 inferencia para evitar latencia"""
        _ = self.session.run(self.output_names, {self.input_name(): input_tensor})

    def infer(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        return list(self.session.run(self.output_names, {self.input_name(): input_tensor}))