from __future__ import annotations

import argparse
import logging
from pathlib import Path

from balsa_vision.config.loader import load_settings
from balsa_vision.storage.layout import make_results_layout


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="balsa-infer", description="PoC inferencia defectos balsa (Paso 1).")
    p.add_argument("--source", choices=["camera", "folder"], required=True, help="Fuente de entrada.")
    p.add_argument("--folder", default="", help="Carpeta de imágenes (si --source folder).")
    p.add_argument("--config-dir", default="configs", help="Directorio de configs YAML.")
    p.add_argument("--results-dir", default="", help="Override de results_dir.")
    p.add_argument("--verbose", action="store_true", help="Logging detallado.")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    log = logging.getLogger("balsa_vision")

    settings = load_settings(args.config_dir)
    results_dir = args.results_dir.strip() or settings.app.results_dir
    layout = make_results_layout(results_dir)

    log.info("Loaded config from: %s", Path(args.config_dir).resolve())
    log.info("Mode: %s", args.source)
    log.info("Results layout: %s", layout.root.resolve())

    from balsa_vision.capture.folder import FolderSource
    from balsa_vision.core.pipeline import FolderPipeline
    from balsa_vision.inference.onnx_yolo import OnnxYoloRunner, OnnxYoloConfig
    from balsa_vision.core.pipeline import OnnxDetector
    from balsa_vision.logging_.writers import JsonlWriter
    from balsa_vision.storage.saver import EvidenceSaver

    if args.source == "folder":
        source = FolderSource(folder=args.folder, recursive=False, sort=True)
        runner = OnnxYoloRunner(OnnxYoloConfig(model_path=settings.model.model_path, providers=settings.model.providers))
        saver = EvidenceSaver(layout=layout)
        writer = JsonlWriter(path=str(layout.logs_dir / "panels.jsonl"))
        pipeline = FolderPipeline(
            detector=OnnxDetector(
                runner=runner,
                input_size=settings.model.input_size,
                class_names=[c.name for c in settings.classes.classes],
                conf_thres=settings.thresholds.conf_thres,
            ),
            saver=saver,
            writer=writer,
            save_raw=settings.app.save_raw,
            save_annotated=settings.app.save_annotated,
        )

        count = 0
        for frame in source:
            res = pipeline.process(frame)
            log.info("Processed panel=%s ok=%s detections=%d", res.panel_id, res.ok, len(res.detections))
            count += 1

        log.info("Done. Total processed: %d", count)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())