from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas import Detection


class Detector(ABC):
    model_id: str

    @abstractmethod
    async def detect(
        self,
        image_data: str,
        width: int,
        height: int,
        confidence_threshold: float,
    ) -> list[Detection]:
        """Return detections in source image pixel coordinates."""
