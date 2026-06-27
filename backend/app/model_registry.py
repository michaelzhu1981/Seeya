from __future__ import annotations

import importlib.util
from pathlib import Path

from app.detectors.base import Detector
from app.detectors.demo_detector import DemoDetector
from app.detectors.onnx_detector import OnnxDetector
from app.detectors.torch_detector import TorchMpsDetector
from app.schemas import ModelInfo

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


class ModelRegistry:
    def __init__(self) -> None:
        self.models = self._discover_models()
        self.selected_model_id = self._recommended_model().id
        self._detector_cache: dict[str, Detector] = {}

    def _discover_models(self) -> list[ModelInfo]:
        onnx_path = MODELS_DIR / "yolo-nano.onnx"
        has_onnxruntime = importlib.util.find_spec("onnxruntime") is not None
        has_torch = importlib.util.find_spec("torch") is not None
        has_ultralytics = importlib.util.find_spec("ultralytics") is not None
        coreml_available = False
        coreml_error: str | None = None
        if has_onnxruntime:
            try:
                import onnxruntime

                coreml_available = "CoreMLExecutionProvider" in onnxruntime.get_available_providers()
            except Exception as exc:
                coreml_available = False
                coreml_error = str(exc)

        real_onnx_available = onnx_path.exists() and has_onnxruntime
        torch_mps_available = self._torch_mps_available(has_torch, has_ultralytics)
        coreml_model_available, coreml_startup_error = self._coreml_model_available(
            onnx_path,
            has_onnxruntime,
            coreml_available,
        )
        coreml_error = coreml_error or coreml_startup_error

        return [
            ModelInfo(
                id="yolo-nano-onnx-cpu",
                name="YOLO Nano ONNX CPU",
                runtime="onnx-cpu",
                size="nano",
                inputSize=416,
                available=real_onnx_available,
                recommended=real_onnx_available,
                unavailableReason=self._onnx_unavailable_reason(onnx_path, has_onnxruntime),
            ),
            ModelInfo(
                id="demo-local-detector",
                name="Demo Local Detector",
                runtime="demo",
                size="demo",
                inputSize=416,
                available=True,
                recommended=not real_onnx_available,
                unavailableReason=None,
            ),
            ModelInfo(
                id="yolo-nano-torch-mps",
                name="YOLO Nano PyTorch MPS",
                runtime="torch-mps",
                size="nano",
                inputSize=416,
                available=torch_mps_available,
                recommended=False,
                unavailableReason=self._torch_unavailable_reason(has_torch, has_ultralytics, torch_mps_available),
            ),
            ModelInfo(
                id="yolo-nano-onnx-coreml",
                name="YOLO Nano ONNX CoreML EP",
                runtime="onnx-coreml",
                size="nano",
                inputSize=416,
                available=coreml_model_available,
                recommended=False,
                unavailableReason=self._coreml_unavailable_reason(
                    onnx_path,
                    has_onnxruntime,
                    coreml_available,
                    coreml_error,
                ),
            ),
        ]

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
    def _torch_mps_available(has_torch: bool, has_ultralytics: bool) -> bool:
        if not (has_torch and has_ultralytics):
            return False
        try:
            import torch

            return bool(torch.backends.mps.is_available())
        except Exception:
            return False

    @staticmethod
    def _torch_unavailable_reason(has_torch: bool, has_ultralytics: bool, available: bool) -> str | None:
        if available:
            return None
        if not has_torch:
            return "torch is not installed"
        if not has_ultralytics:
            return "ultralytics is not installed"
        return "MPS is not available"

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
        elif model.id == "yolo-nano-onnx-cpu":
            detector = OnnxDetector(MODELS_DIR / "yolo-nano.onnx")
        elif model.id == "yolo-nano-torch-mps":
            detector = TorchMpsDetector()
        elif model.id == "yolo-nano-onnx-coreml":
            detector = OnnxDetector(
                MODELS_DIR / "yolo-nano.onnx",
                model_id="yolo-nano-onnx-coreml",
                providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
            )
        else:
            raise KeyError(model.id)

        self._detector_cache[model.id] = detector
        return detector
