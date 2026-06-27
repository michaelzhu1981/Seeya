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

The recommended high-accuracy detector is `yolo-small-onnx-cpu` with a 640 px
input. If that model is not available, the backend falls back to
`demo-local-detector` so the camera and WebSocket loop remain testable.

The ONNX CPU models need:

- the ONNX Runtime package installed in the backend virtual environment
- a YOLO ONNX model file at `backend/models/yolo-small.onnx` for the recommended high-accuracy path

Install the runtime and export tools inside the backend virtual environment:

```bash
cd backend
source .venv/bin/activate
python -m pip install onnxruntime ultralytics
```

Export the recommended small YOLO model to ONNX and move it to the path expected
by the backend:

```bash
mkdir -p models
yolo export model=yolov8s.pt format=onnx imgsz=640
mv yolov8s.onnx models/yolo-small.onnx
```

Optional higher-accuracy YOLO11 models can be exported to the additional model
paths surfaced in the UI:

```bash
yolo export model=yolo11l.pt format=onnx imgsz=640
mv yolo11l.onnx models/yolo11l.onnx

yolo export model=yolo11x.pt format=onnx imgsz=640
mv yolo11x.onnx models/yolo11x.onnx
```

`onnxruntime` is only the inference engine. It does not include a YOLO model or
trained weights. The exported `models/yolo-small.onnx` file is the model artifact
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

With `backend/models/yolo-small.onnx` and `onnxruntime` present, the selected
model is `yolo-small-onnx-cpu`. Without it, the selected model is
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

- YOLO Small ONNX CPU: `backend/models/yolo-small.onnx` + `onnxruntime`
- YOLO11 Large ONNX CPU: `backend/models/yolo11l.onnx` + `onnxruntime`
- YOLO11 X ONNX CPU: `backend/models/yolo11x.onnx` + `onnxruntime`
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

## License

This project is licensed under the MIT License.
