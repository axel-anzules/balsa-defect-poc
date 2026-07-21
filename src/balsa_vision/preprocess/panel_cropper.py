# preprocess/panel_cropper.py
"""
Detección, enderezado y recorte del panel de balsa dentro del frame
capturado por la cámara ADLINK (lente ojo de pez + marco de espuma).

Implementa la estrategia híbrida definida en ADR-010:
    1. Detección del panel por umbral de saturación en espacio HSV
       (el panel es cálido/saturado; el marco de espuma y el fondo son
       desaturados).
    2. Cálculo del rectángulo mínimo rotado (minAreaRect) que envuelve
       el panel, capturando su inclinación real.
    3. Rotación de la imagen completa para enderezar el panel (solo
       rotación afín, no homografía — ver justificación en ADR-010).
    4. Recorte con margen de seguridad configurable (ADR-011).
    5. Transformación consistente de las bounding boxes YOLO a través
       de la misma rotación + recorte, con clipping/descarte explícito
       de cajas que queden parcialmente fuera del área recortada.

Toda la geometría se valida contra rangos calibrados estadísticamente
sobre el propio dataset (ver panel_crop_dataset.py), nunca contra
umbrales fijos asumidos a priori.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from balsa_vision.dataset_analysis.common import BoundingBox

logger = logging.getLogger(__name__)

# Rango de Hue (OpenCV, 0-179) que identifica el panel de balsa (tono
# cálido amarillo-anaranjado), calibrado empíricamente contra la
# distribución real del dataset (ver hue_calibration.py). Descarta el
# criterio de saturación como discriminante principal (ver ADR-015):
# tras el contrast stretching de Roboflow, panel y marco pueden tener
# saturación similar, pero su matiz (identidad de color) permanece
# claramente separado.
DEFAULT_HUE_RANGE: tuple[int, int] = (10, 45)

# Saturación mínima solo para excluir píxeles sin color definible
# (negro puro del vignette del lente), no para discriminar panel/marco.
DEFAULT_MIN_SATURATION = 20

# Chequeos de sanidad física (ADR-016): independientes de la
# calibración estadística por percentiles, para blindar contra
# detecciones geométricamente imposibles (ej. rectángulos que exceden
# el frame o ángulos de rotación no plausibles para una cámara fija).
MAX_PLAUSIBLE_ROTATION_DEGREES = 15.0
MAX_FRAME_EXCESS_RATIO = 1.05  # el rect detectado no puede exceder el frame en más de 5%

# Fracción mínima de intersección entre una bbox original y el área de
# recorte para conservarla (clippeada). Por debajo de este umbral, la
# caja se descarta en vez de conservarse truncada casi en su totalidad.
MIN_BOX_INTERSECTION_RATIO = 0.5

@dataclass
class PanelRect:
    """Rectángulo rotado que representa el panel detectado."""
    center: tuple[float, float]
    size: tuple[float, float]  # (width, height), width siempre >= height tras normalizar
    angle: float  # grados, ángulo de rotación necesario para enderezar

    @property
    def area(self) -> float:
        return self.size[0] * self.size[1]

    @property
    def aspect_ratio(self) -> float:
        return self.size[0] / self.size[1] if self.size[1] > 0 else 0.0


@dataclass
class PanelDetectionResult:
    """Resultado de intentar detectar el panel en una imagen."""
    success: bool
    panel_rect: PanelRect | None = None
    area_ratio: float | None = None  # área del panel / área total del frame
    failure_reason: str | None = None


@dataclass
class CropResult:
    """Resultado final del proceso de crop + transformación de boxes."""
    success: bool
    cropped_image: np.ndarray | None = None
    transformed_boxes: list[BoundingBox] = field(default_factory=list)
    dropped_box_count: int = 0
    failure_reason: str | None = None


def _normalize_rect_angle(rect: tuple) -> PanelRect:
    """
    cv2.minAreaRect puede devolver (w, h, angle) en cualquier orientación.
    Se normaliza para que 'width' sea siempre el lado más largo, ajustando
    el ángulo correspondientemente, ya que sabemos que el panel real es
    más ancho que alto (aspect ratio ~4:3, confirmado en Fase 1).
    """
    (cx, cy), (w, h), angle = rect
    if w < h:
        w, h = h, w
        angle += 90.0
    return PanelRect(center=(cx, cy), size=(w, h), angle=angle)


def detect_panel(
    image: np.ndarray,
    hue_range: tuple[int, int] = DEFAULT_HUE_RANGE,
    min_saturation: int = DEFAULT_MIN_SATURATION,
) -> PanelDetectionResult:
    """
    Detecta el panel dentro del frame usando el canal de Hue (matiz) en
    espacio HSV, explotando que el panel de balsa (cálido, Hue~20-31)
    es cromáticamente distinto del marco de espuma (frío, Hue~124-147),
    señal robusta frente a variaciones de brillo/exposición entre
    sesiones (ver ADR-015).

    Incluye validación de sanidad física (ADR-016): rechaza detecciones
    cuyo rectángulo exceda el frame o cuyo ángulo de rotación no sea
    plausible para una cámara montada en soporte fijo, antes de que
    lleguen a la calibración estadística por percentiles (que por sí
    sola no protege contra sesgos sistemáticos).
    """
    h, w = image.shape[:2]
    frame_area = float(h * w)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation = hsv[:, :, 0], hsv[:, :, 1]

    hue_mask = cv2.inRange(hue, hue_range[0], hue_range[1])
    sat_mask = cv2.inRange(saturation, min_saturation, 255)
    mask = cv2.bitwise_and(hue_mask, sat_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return PanelDetectionResult(success=False, failure_reason="no_contours_found")

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < frame_area * 0.01:
        return PanelDetectionResult(success=False, failure_reason="largest_contour_too_small")

    rect = cv2.minAreaRect(largest_contour)
    panel_rect = _normalize_rect_angle(rect)
    area_ratio = panel_rect.area / frame_area

    # --- Validación de sanidad física (ADR-016) ---
    if area_ratio > MAX_FRAME_EXCESS_RATIO:
        return PanelDetectionResult(
            success=False,
            failure_reason=f"detected_rect_exceeds_frame (area_ratio={area_ratio:.3f})",
        )

    # El ángulo normalizado puede caer en cualquier múltiplo de 90°;
    # se evalúa la distancia angular al múltiplo de 90 más cercano,
    # que representa la inclinación real respecto a "derecho".
    angle_mod = panel_rect.angle % 90.0
    deviation_from_axis = min(angle_mod, 90.0 - angle_mod)
    if deviation_from_axis > MAX_PLAUSIBLE_ROTATION_DEGREES:
        return PanelDetectionResult(
            success=False,
            failure_reason=(
                f"implausible_rotation_angle (deviation={deviation_from_axis:.1f}°, "
                f"max_allowed={MAX_PLAUSIBLE_ROTATION_DEGREES}°)"
            ),
        )

    return PanelDetectionResult(success=True, panel_rect=panel_rect, area_ratio=area_ratio)


def validate_panel_geometry(
    detection: PanelDetectionResult,
    area_ratio_range: tuple[float, float],
    aspect_ratio_range: tuple[float, float],
) -> tuple[bool, str | None]:
    """
    Valida un panel detectado contra rangos calibrados estadísticamente
    sobre el dataset completo (ver panel_crop_dataset.py). No usa
    umbrales asumidos a priori.
    """
    if not detection.success:
        return False, detection.failure_reason

    assert detection.panel_rect is not None and detection.area_ratio is not None

    min_area, max_area = area_ratio_range
    if not (min_area <= detection.area_ratio <= max_area):
        return False, (
            f"area_ratio {detection.area_ratio:.4f} fuera de rango "
            f"[{min_area:.4f}, {max_area:.4f}]"
        )

    min_ar, max_ar = aspect_ratio_range
    if not (min_ar <= detection.panel_rect.aspect_ratio <= max_ar):
        return False, (
            f"aspect_ratio {detection.panel_rect.aspect_ratio:.3f} fuera de rango "
            f"[{min_ar:.3f}, {max_ar:.3f}]"
        )

    return True, None


def _rotate_image_and_get_transform(
    image: np.ndarray, center: tuple[float, float], angle: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rota la imagen completa alrededor de 'center' por 'angle' grados,
    expandiendo el canvas para no perder contenido en las esquinas
    (técnica estándar: recalcular dimensiones tras rotación y ajustar
    la componente de traslación de la matriz de transformación).

    Devuelve la imagen rotada y la matriz de transformación 2x3 usada,
    para poder aplicar la misma transformación a las bounding boxes.
    """
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Ajusta la traslación para centrar el contenido en el nuevo canvas.
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    return rotated, M


def _transform_point(point: tuple[float, float], M: np.ndarray) -> tuple[float, float]:
    """Aplica una matriz de transformación afín 2x3 a un punto (x, y)."""
    x, y = point
    new_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    new_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    return new_x, new_y


def _box_to_pixel_corners(box: BoundingBox, img_w: int, img_h: int) -> np.ndarray:
    """Convierte una BoundingBox normalizada YOLO a sus 4 esquinas en píxeles absolutos."""
    x1 = (box.x_center - box.width / 2) * img_w
    y1 = (box.y_center - box.height / 2) * img_h
    x2 = (box.x_center + box.width / 2) * img_w
    y2 = (box.y_center + box.height / 2) * img_h
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


def crop_panel(
    image: np.ndarray,
    boxes: list[BoundingBox],
    panel_rect: PanelRect,
    margin_ratio: float,
    min_box_intersection_ratio: float = MIN_BOX_INTERSECTION_RATIO,
) -> CropResult:
    """
    Ejecuta el pipeline completo: rota la imagen para enderezar el panel,
    calcula la región de recorte con margen de seguridad, recorta, y
    transforma/filtra las bounding boxes de forma consistente.
    """
    img_h, img_w = image.shape[:2]

    rotated_image, M = _rotate_image_and_get_transform(image, panel_rect.center, panel_rect.angle)
    rot_h, rot_w = rotated_image.shape[:2]

    # Centro del panel en el nuevo canvas rotado (el centro no se mueve
    # respecto al panel, solo el canvas se expande a su alrededor).
    panel_center_rotated = _transform_point(panel_rect.center, M)

    half_w = (panel_rect.size[0] / 2) * (1 + margin_ratio)
    half_h = (panel_rect.size[1] / 2) * (1 + margin_ratio)

    crop_x1 = int(round(panel_center_rotated[0] - half_w))
    crop_y1 = int(round(panel_center_rotated[1] - half_h))
    crop_x2 = int(round(panel_center_rotated[0] + half_w))
    crop_y2 = int(round(panel_center_rotated[1] + half_h))

    # Clip contra los límites reales del canvas rotado.
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(rot_w, crop_x2)
    crop_y2 = min(rot_h, crop_y2)

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return CropResult(success=False, failure_reason="degenerate_crop_region")

    cropped_image = rotated_image[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    transformed_boxes: list[BoundingBox] = []
    dropped_count = 0

    for box in boxes:
        original_corners = _box_to_pixel_corners(box, img_w, img_h)
        original_area = box.width * box.height * img_w * img_h

        rotated_corners = np.array([_transform_point(tuple(pt), M) for pt in original_corners])

        # Caja axis-aligned mínima que envuelve las 4 esquinas rotadas
        # (una caja rotada deja de ser axis-aligned; YOLO exige que lo sea).
        rx1, ry1 = rotated_corners.min(axis=0)
        rx2, ry2 = rotated_corners.max(axis=0)

        # Trasladar al sistema de coordenadas del recorte.
        cx1 = rx1 - crop_x1
        cy1 = ry1 - crop_y1
        cx2 = rx2 - crop_x1
        cy2 = ry2 - crop_y1

        # Intersección con el área de recorte.
        ix1 = max(cx1, 0)
        iy1 = max(cy1, 0)
        ix2 = min(cx2, crop_w)
        iy2 = min(cy2, crop_h)

        if ix2 <= ix1 or iy2 <= iy1:
            dropped_count += 1
            continue

        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        rotated_box_area = max((cx2 - cx1) * (cy2 - cy1), 1e-6)
        intersection_ratio = intersection_area / rotated_box_area

        if intersection_ratio < min_box_intersection_ratio:
            dropped_count += 1
            continue

        new_xc = ((ix1 + ix2) / 2) / crop_w
        new_yc = ((iy1 + iy2) / 2) / crop_h
        new_w = (ix2 - ix1) / crop_w
        new_h = (iy2 - iy1) / crop_h

        transformed_boxes.append(
            BoundingBox(class_id=box.class_id, x_center=new_xc, y_center=new_yc, width=new_w, height=new_h)
        )

    return CropResult(
        success=True,
        cropped_image=cropped_image,
        transformed_boxes=transformed_boxes,
        dropped_box_count=dropped_count,
    )