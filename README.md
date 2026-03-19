# Balsa Defect PoC (NEON-202B / Jetson)

PoC industrial para detección de defectos en paneles de balsa (2x4 ft) con cámara ADLINK NEON-202B y modelo YOLO exportado a ONNX.

## Estructura
- `src/`: código principal (pipeline, captura, inferencia, postproceso, logging)
- `configs/`: configuración YAML (clases, umbrales, modelo, fuentes)
- `results/YYYY-MM-DD/`: evidencia y logs por día (raw/ annotated/ logs/)

## Setup PC (Windows)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt"