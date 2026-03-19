from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from balsa_vision.capture.base import Frame
from balsa_vision.core.types import Detection, PanelResult
from balsa_vision.inference.onnx_yolo import OnnxYoloRunner
from balsa_vision.preprocess.transforms import bgr_to_nchw_float, letterbox_bgr
from balsa_vision.postprocess.parse import parse_simple_out
from balsa_vision.logging_.writers import JsonlWriter
from balsa_vision.storage.saver import EvidenceSaver
from balsa_vision.viz.annotate import draw_detections


#@dataclass
#class StubDetector:
#    """Detector temporal: no hace inferencia real, devuelve lista vacía."""
#    def detect(self, frame: Frame) -> List[Detection]:
#        return []

@dataclass
class OnnxDetector:
    runner: OnnxYoloRunner
    input_size: int
    class_names: Sequence[str]
    conf_thres: float
    def detect(self, frame: Frame) -> List[Detection]:
        # Preprocess (letterbox to tensor)
        lb, meta = letterbox_bgr(frame.image, self.input_size)
        x = bgr_to_nchw_float(lb)

        outputs = self.runner.infer(x)

        # Postprocess minimo
        dets = parse_simple_out(outputs, meta, self.class_names, self.conf_thres)
        return dets

@dataclass
class FolderPipeline:
    detector: OnnxDetector
    saver: EvidenceSaver
    writer: JsonlWriter
    save_raw: bool
    save_annotated: bool

    def process(self, frame: Frame) -> PanelResult:
        detections = self.detector.detect(frame)

        ok = True if len(detections) == 0 else False
        reason = "no_defects_detected_stub" if ok else "defects_detected_stub"

        filename = f"{frame.frame_id}_{frame.timestamp_ms}.jpg"

        raw_path = ""
        if self.save_raw:
            raw_path = self.saver.save_raw(filename, frame.image)

        annotated_path = ""
        if self.save_annotated:
            annotated = draw_detections(frame.image, detections)
            annotated_path = self.saver.save_annotated(filename, annotated)

        result = PanelResult(
            panel_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            ok=ok,
            reason=reason,
            detections=detections,
            raw_path=raw_path,
            annotated_path=annotated_path,
        )

        self.writer.append(result)
        return result