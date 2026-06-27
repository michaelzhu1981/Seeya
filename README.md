# Seeya

Seeya is a local real-time object detection workbench. The first phase uses a
browser camera feed, a React/Vite frontend, and a local FastAPI backend over
WebSocket.

## Repository Layout

```text
backend/
  app/
    main.py
    model_registry.py
    schemas.py
    detectors/
  models/
    .gitkeep
  scripts/
    benchmark.py
frontend/
  src/
```

## Backend Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8010
```

## YOLO Model Setup

The recommended detector is `yolo-nano-onnx-cpu`. When the ONNX model file and
runtime are available, the backend selects it by default and returns real YOLO
detections. If either dependency is missing, the backend falls back to
`demo-local-detector` so the camera and WebSocket loop remain testable.

The ONNX CPU model needs:

- the ONNX Runtime package installed in the backend virtual environment
- a YOLO ONNX model file at `backend/models/yolo-nano.onnx`

Install the runtime and export tools inside the backend virtual environment:

```bash
cd backend
source .venv/bin/activate
python -m pip install onnxruntime ultralytics
```

Export the Ultralytics nano YOLO model to ONNX and move it to the path expected
by the backend:

```bash
mkdir -p models
yolo export model=yolov8n.pt format=onnx imgsz=416
mv yolov8n.onnx models/yolo-nano.onnx
```

`onnxruntime` is only the inference engine. It does not include a YOLO model or
trained weights. The exported `models/yolo-nano.onnx` file is the model artifact
that the backend loads.

## One-command Development

From the repository root:

```bash
./run-dev.sh
```

The script installs missing backend/frontend dependencies, starts the FastAPI
backend on `http://127.0.0.1:8010`, starts the Vite frontend on
`http://127.0.0.1:5173`, and stops both servers when you press `Ctrl+C`.

The backend exposes:

- `GET /health`
- `GET /models`
- `POST /models/select`
- `WS /ws/detect`

With `backend/models/yolo-nano.onnx` and `onnxruntime` present, the selected
model is `yolo-nano-onnx-cpu`. Without them, the selected model is
`demo-local-detector`.

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL in a browser and allow camera access. The frontend expects the
backend at `http://127.0.0.1:8010` by default. Override it with:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8010 npm run dev
```

## Model Runtime Notes

Phase one runs locally by default. Docker is intentionally not the primary
camera and Mac runtime benchmark path because browser camera permissions, MPS,
CoreML Execution Provider, and Apple ANE access are more reliable in the native
development environment.

Optional runtimes appear in the model list only when their dependencies and
local files are available:

- YOLO Nano ONNX CPU: `backend/models/yolo-nano.onnx` + `onnxruntime`
- PyTorch MPS: `torch`, `ultralytics`, and MPS availability
- ONNX Runtime CoreML EP: `onnxruntime` with `CoreMLExecutionProvider`

Camera frames are sent to the backend for inference and are not saved to disk.

## Verification

Backend unit tests:

```bash
cd backend
source .venv/bin/activate
python -m pytest
```

Frontend build:

```bash
cd frontend
npm run build
```

Benchmark scaffold:

```bash
cd backend
source .venv/bin/activate
python scripts/benchmark.py
```
