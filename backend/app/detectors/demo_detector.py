from __future__ import annotations

import hashlib

from app.detectors.base import Detector
from app.schemas import Box, Detection


class DemoDetector(Detector):
    """Deterministic local detector used to validate the real-time app loop."""

    model_id = "demo-local-detector"

    async def detect(
        self,
        image_data: str,
        width: int,
        height: int,
        confidence_threshold: float,
    ) -> list[Detection]:
        digest = hashlib.sha1(image_data[:4096].encode("utf-8")).digest()
        drift = digest[0] / 255
        candidates = [
            Detection(
                label="person",
                confidence=0.86,
                box=Box(
                    x=width * (0.12 + drift * 0.04),
                    y=height * 0.16,
                    width=width * 0.26,
                    height=height * 0.58,
                ),
            ),
            Detection(
                label="cup",
                confidence=0.68,
                box=Box(
                    x=width * 0.58,
                    y=height * (0.48 + drift * 0.03),
                    width=width * 0.12,
                    height=height * 0.18,
                ),
            ),
            Detection(
                label="keyboard",
                confidence=0.61,
                box=Box(
                    x=width * 0.36,
                    y=height * 0.72,
                    width=width * 0.42,
                    height=height * 0.12,
                ),
            ),
        ]
        return [item for item in candidates if item.confidence >= confidence_threshold]
