from __future__ import annotations

import importlib.util
from pathlib import Path

from app.detectors.base import Detector
from app.detectors.demo_detector import DemoDetector
from app.detectors.onnx_detector import OnnxDetector
from app.schemas import ModelInfo

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


ONNX_MODEL_SPECS = [
    {
        "id": "yolo-small-onnx-cpu",
        "name": "YOLO Small ONNX CPU",
        "runtime": "onnx-cpu",
        "size": "small",
        "input_size": 640,
        "path": MODELS_DIR / "yolo-small.onnx",
    },
    {
        "id": "yolo11l-onnx-cpu",
        "name": "YOLO11 Large ONNX CPU",
        "runtime": "onnx-cpu",
        "size": "large",
        "input_size": 640,
        "path": MODELS_DIR / "yolo11l.onnx",
    },
    {
        "id": "yolo11x-onnx-cpu",
        "name": "YOLO11 X ONNX CPU",
        "runtime": "onnx-cpu",
        "size": "xlarge",
        "input_size": 640,
        "path": MODELS_DIR / "yolo11x.onnx",
    },
]


class ModelRegistry:
    def __init__(self) -> None:
        self.models = self._discover_models()
        self.selected_model_id = self._recommended_model().id
        self._detector_cache: dict[str, Detector] = {}

    def _discover_models(self) -> list[ModelInfo]:
        has_onnxruntime = importlib.util.find_spec("onnxruntime") is not None
        coreml_available = False
        coreml_error: str | None = None
        if has_onnxruntime:
            try:
                import onnxruntime

                coreml_available = "CoreMLExecutionProvider" in onnxruntime.get_available_providers()
            except Exception as exc:
                coreml_available = False
                coreml_error = str(exc)

        recommended_onnx_id = self._recommended_onnx_id(has_onnxruntime)
        real_onnx_available = recommended_onnx_id is not None
        models: list[ModelInfo] = []

        for spec in ONNX_MODEL_SPECS:
            model_path = spec["path"]
            available = model_path.exists() and has_onnxruntime
            models.append(
                ModelInfo(
                    id=spec["id"],
                    name=spec["name"],
                    runtime=spec["runtime"],
                    size=spec["size"],
                    inputSize=spec["input_size"],
                    available=available,
                    recommended=spec["id"] == recommended_onnx_id,
                    unavailableReason=self._onnx_unavailable_reason(model_path, has_onnxruntime),
                ),
            )

        for spec in ONNX_MODEL_SPECS:
            model_path = spec["path"]
            coreml_model_available, coreml_startup_error = self._coreml_model_available(
                model_path,
                has_onnxruntime,
                coreml_available,
            )
            model_coreml_error = coreml_error or coreml_startup_error
            models.append(
                ModelInfo(
                    id=spec["id"].replace("-cpu", "-coreml"),
                    name=f"{spec['name'].removesuffix(' CPU')} CoreML EP",
                    runtime="onnx-coreml",
                    size=spec["size"],
                    inputSize=spec["input_size"],
                    available=coreml_model_available,
                    recommended=False,
                    unavailableReason=self._coreml_unavailable_reason(
                        model_path,
                        has_onnxruntime,
                        coreml_available,
                        model_coreml_error,
                    ),
                )
            )

        models.append(
            ModelInfo(
                id="demo-local-detector",
                name="Demo Local Detector",
                runtime="demo",
                size="demo",
                inputSize=416,
                available=True,
                recommended=not real_onnx_available,
                unavailableReason=None,
            )
        )
        return models

    @staticmethod
    def _recommended_onnx_id(has_onnxruntime: bool) -> str | None:
        if not has_onnxruntime:
            return None
        for spec in ONNX_MODEL_SPECS:
            if spec["path"].exists():
                return spec["id"]
        return None

    def _recommended_model(self) -> ModelInfo:
        for model in self.models:
            if model.available and model.recommended:
                return model
        for model in self.models:
            if model.available:
                return model
        raise RuntimeError("No detector models are available")

    @staticmethod
    def _onnx_unavailable_reason(onnx_path: Path, has_onnxruntime: bool) -> str | None:
        if not onnx_path.exists():
            return f"Missing model file: {onnx_path}"
        if not has_onnxruntime:
            return "onnxruntime is not installed"
        return None

    @staticmethod
    def _coreml_model_available(
        onnx_path: Path,
        has_onnxruntime: bool,
        coreml_available: bool,
    ) -> tuple[bool, str | None]:
        if not (onnx_path.exists() and has_onnxruntime and coreml_available):
            return False, None
        try:
            import onnxruntime

            session = onnxruntime.InferenceSession(
                str(onnx_path),
                providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
            )
            return "CoreMLExecutionProvider" in session.get_providers(), None
        except Exception as exc:
            return False, str(exc)

    @staticmethod
    def _coreml_unavailable_reason(
        onnx_path: Path,
        has_onnxruntime: bool,
        coreml_available: bool,
        coreml_error: str | None,
    ) -> str | None:
        if not onnx_path.exists():
            return f"Missing model file: {onnx_path}"
        if not has_onnxruntime:
            return "onnxruntime is not installed"
        if not coreml_available:
            return "CoreMLExecutionProvider is not available"
        if coreml_error:
            return f"CoreMLExecutionProvider startup check failed: {coreml_error}"
        return None

    def list_models(self) -> list[ModelInfo]:
        return self.models

    def selected_model(self) -> ModelInfo:
        return self.require_model(self.selected_model_id)

    def require_model(self, model_id: str) -> ModelInfo:
        for model in self.models:
            if model.id == model_id:
                return model
        raise KeyError(model_id)

    def select_model(self, model_id: str) -> ModelInfo:
        model = self.require_model(model_id)
        if not model.available:
            raise ValueError(model.unavailableReason or "Model is unavailable")
        self.selected_model_id = model.id
        return model

    def detector_for(self, model_id: str | None = None) -> Detector:
        target_id = model_id or self.selected_model_id
        model = self.select_model(target_id)
        cached = self._detector_cache.get(model.id)
        if cached:
            return cached

        if model.id == "demo-local-detector":
            detector: Detector = DemoDetector()
        elif model.id in {spec["id"] for spec in ONNX_MODEL_SPECS}:
            spec = self._onnx_spec_for(model.id)
            detector = OnnxDetector(
                spec["path"],
                model_id=spec["id"],
                input_size=spec["input_size"],
            )
        elif model.id in {spec["id"].replace("-cpu", "-coreml") for spec in ONNX_MODEL_SPECS}:
            cpu_model_id = model.id.replace("-coreml", "-cpu")
            spec = self._onnx_spec_for(cpu_model_id)
            detector = OnnxDetector(
                spec["path"],
                model_id=model.id,
                providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
                input_size=spec["input_size"],
            )
        else:
            raise KeyError(model.id)

        self._detector_cache[model.id] = detector
        return detector

    @staticmethod
    def _onnx_spec_for(model_id: str) -> dict[str, object]:
        for spec in ONNX_MODEL_SPECS:
            if spec["id"] == model_id:
                return spec
        raise KeyError(model_id)
