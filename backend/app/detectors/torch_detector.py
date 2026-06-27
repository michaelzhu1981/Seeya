from __future__ import annotations

from app.detectors.demo_detector import DemoDetector


class TorchMpsDetector(DemoDetector):
    model_id = "yolo-nano-torch-mps"

    def __init__(self) -> None:
        try:
            import torch
            import ultralytics  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("torch and ultralytics are required") from exc
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available on this Mac")
