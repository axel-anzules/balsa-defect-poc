# dataset_analysis/visual_sampling.py
"""
Genera grids de muestreo visual para validación humana de los hallazgos
estadísticos de los reportes anteriores. No reemplaza el juicio humano:
su propósito es acercar la evidencia visual necesaria para decidir si los
"casos problemáticos" detectados estadísticamente (blur, duplicados,
bboxes extremas) son problemas reales o artefactos de las métricas
usadas.

Genera 5 grids independientes:
    1. sample_random_annotated.png   -> muestra aleatoria con bboxes dibujadas
    2. sample_smallest_bboxes.png    -> las N cajas más pequeñas (posibles errores de etiquetado)
    3. sample_largest_bboxes.png     -> las N cajas más grandes (posibles errores de etiquetado)
    4. sample_blurriest.png          -> las N imágenes con menor nitidez
    5. sample_duplicate_group_XX.png -> un grid por cada uno de los K primeros grupos de duplicados

Uso:
    python -m dataset_analysis.visual_sampling \
        --root-dir /ruta/al/dataset \
        --ndjson /ruta/al/archivo.ndjson \
        --duplicates-report ./reports/duplicates_report.json \
        --output-dir ./reports/visual_samples \
        --n-samples 12
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from scripts.dataset_analysis.common import DatasetImage, DatasetInfo, load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

BOX_COLOR = (0, 255, 0)  # BGR verde
BOX_THICKNESS = 3


def draw_boxes(image: np.ndarray, dataset_image: DatasetImage, class_names: dict[int, str]) -> np.ndarray:
    """Dibuja las bounding boxes anotadas sobre una copia de la imagen (formato YOLO -> píxeles)."""
    annotated = image.copy()
    h, w = annotated.shape[:2]
    for box in dataset_image.boxes:
        x1 = int((box.x_center - box.width / 2) * w)
        y1 = int((box.y_center - box.height / 2) * h)
        x2 = int((box.x_center + box.width / 2) * w)
        y2 = int((box.y_center + box.height / 2) * h)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        label = f"{class_names[box.class_id]} ({box.area_ratio*100:.3f}%)"
        cv2.putText(annotated, label, (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2)
    return annotated


def save_grid(images: list[np.ndarray], titles: list[str], output_path: Path, ncols: int = 4) -> None:
    """Guarda una lista de imágenes (BGR) como un grid con títulos individuales."""
    if not images:
        logger.warning("No hay imágenes para generar el grid: %s", output_path.name)
        return
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = np.array(axes).reshape(-1)

    for ax, img, title in zip(axes_flat, images, titles):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    for ax in axes_flat[len(images):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Grid guardado: %s", output_path)


def sample_random_annotated(dataset: DatasetInfo, n: int, output_dir: Path) -> None:
    """Muestra aleatoria de imágenes CON defectos, anotadas, para revisión general de calidad de etiquetado."""
    positives = [img for img in dataset.images if not img.is_negative]
    sample = random.sample(positives, min(n, len(positives)))

    images, titles = [], []
    for dimg in sample:
        img = cv2.imread(str(dimg.image_path))
        images.append(draw_boxes(img, dimg, dataset.class_names))
        titles.append(dimg.image_path.name)

    save_grid(images, titles, output_dir / "sample_random_annotated.png")


def sample_extreme_bboxes(dataset: DatasetInfo, n: int, output_dir: Path, largest: bool) -> None:
    """
    Muestra las N cajas más pequeñas o más grandes de todo el dataset,
    para revisión manual de posibles errores de etiquetado (cajas
    degeneradas o excesivamente grandes).
    """
    entries: list[tuple[float, DatasetImage, int]] = []
    for dimg in dataset.images:
        for idx, box in enumerate(dimg.boxes):
            entries.append((box.area_ratio, dimg, idx))

    entries.sort(key=lambda x: x[0], reverse=largest)
    selected = entries[:n]

    images, titles = [], []
    for area_ratio, dimg, box_idx in selected:
        img = cv2.imread(str(dimg.image_path))
        images.append(draw_boxes(img, dimg, dataset.class_names))
        titles.append(f"{dimg.image_path.name}\narea={area_ratio*100:.4f}%")

    suffix = "largest" if largest else "smallest"
    save_grid(images, titles, output_dir / f"sample_{suffix}_bboxes.png")


def sample_blurriest(dataset: DatasetInfo, n: int, output_dir: Path) -> None:
    """Muestra las N imágenes con menor nitidez (varianza del Laplaciano), sin bboxes."""
    scored: list[tuple[float, DatasetImage]] = []
    for dimg in dataset.images:
        img = cv2.imread(str(dimg.image_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        scored.append((sharpness, dimg))

    scored.sort(key=lambda x: x[0])
    selected = scored[:n]

    images, titles = [], []
    for sharpness, dimg in selected:
        img = cv2.imread(str(dimg.image_path))
        images.append(img)
        titles.append(f"{dimg.image_path.name}\nsharpness={sharpness:.1f}")

    save_grid(images, titles, output_dir / "sample_blurriest.png")


def sample_duplicate_groups(
    duplicates_report_path: Path, output_dir: Path, n_groups: int, max_per_group: int
) -> None:
    """
    Muestra visualmente los primeros N grupos reportados como near-duplicate,
    para determinar si son duplicados reales o falsos positivos causados
    por fondo dominante (ver Hallazgo 4 del ADR).
    """
    with duplicates_report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    groups = report["groups"][:n_groups]
    for group_idx, group_paths in enumerate(groups, start=1):
        images, titles = [], []
        for path_str in group_paths[:max_per_group]:
            img = cv2.imread(path_str)
            if img is None:
                continue
            images.append(img)
            titles.append(Path(path_str).name)
        save_grid(
            images, titles,
            output_dir / f"sample_duplicate_group_{group_idx:02d}.png",
            ncols=min(4, len(images)) or 1,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Muestreo visual de casos problemáticos del dataset.")
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--ndjson", type=Path, default=None)
    parser.add_argument("--data-yaml", type=Path, default=None)
    parser.add_argument("--duplicates-report", type=Path, default=None,
                         help="Ruta a duplicates_report.json (opcional, omite ese grid si no se provee).")
    parser.add_argument("--output-dir", type=Path, default=Path("./reports/visual_samples"))
    parser.add_argument("--n-samples", type=int, default=12)
    parser.add_argument("--n-duplicate-groups", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(root_dir=args.root_dir, ndjson_path=args.ndjson, data_yaml_path=args.data_yaml)

    logger.info("Generando muestra aleatoria anotada...")
    sample_random_annotated(dataset, args.n_samples, args.output_dir)

    logger.info("Generando muestra de bboxes más pequeñas...")
    sample_extreme_bboxes(dataset, args.n_samples, args.output_dir, largest=False)

    logger.info("Generando muestra de bboxes más grandes...")
    sample_extreme_bboxes(dataset, args.n_samples, args.output_dir, largest=True)

    logger.info("Generando muestra de imágenes menos nítidas...")
    sample_blurriest(dataset, args.n_samples, args.output_dir)

    if args.duplicates_report is not None and args.duplicates_report.exists():
        logger.info("Generando muestras de grupos de duplicados...")
        sample_duplicate_groups(
            args.duplicates_report, args.output_dir, args.n_duplicate_groups, max_per_group=4
        )
    else:
        logger.info("No se proveyó --duplicates-report; se omite ese grid.")

    logger.info("Muestreo visual completo. Archivos en: %s", args.output_dir)


if __name__ == "__main__":
    main()