# src/balsa_vision/preprocess/image_enhancement.py
"""
Funciones de mejora fotométrica del pipeline de preprocesamiento. 
Separado de panel_cropper.py porque atiende una
responsabilidad distinta: ajuste de contraste/iluminación, no
geometría del panel.

Estas funciones se reutilizan de forma idéntica en:
    - El pipeline de materialización del dataset de entrenamiento
      (clahe_dataset.py, este módulo).
    - El pipeline de inferencia en tiempo real en Jetson,
      para garantizar consistencia entre el dominio de entrenamiento
      y el dominio de producción.

Parámetros calibrados empíricamente sobre el dataset real — ver
clahe_calibration.py para la justificación de estos valores.
"""

from __future__ import annotations

import cv2
import numpy as np

# Calibrados visualmente sobre 9 muestras estratificadas por brillo.
# No modificar sin repetir el proceso de calibración.
# visual — valores distintos pueden introducir artefactos de bloque
# (tile pequeño) o ruido granulado (clipLimit alto).
CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_GRID_SIZE: tuple[int, int] = (8, 8)


def apply_clahe(
    image_bgr: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: tuple[int, int] = CLAHE_TILE_GRID_SIZE,
) -> np.ndarray:
    """
    Aplica CLAHE sobre el canal L (luminancia) del espacio Lab y
    reconstruye la imagen en BGR. Operar sobre L en vez de directamente
    sobre los canales BGR evita distorsionar el balance de color de la
    imagen (relevante porque el Hue del panel es la señal usada en la
    detección geométrica).
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_equalized = clahe.apply(l_channel)

    lab_equalized = cv2.merge([l_equalized, a_channel, b_channel])
    return cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)