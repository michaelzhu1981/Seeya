from __future__ import annotations

import asyncio
import base64
import io
import statistics
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.model_registry import ModelRegistry
from PIL import Image, ImageDraw


def _sample_image_data(width: int, height: int) -> str:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((width * 0.36, height * 0.18, width * 0.64, height * 0.92), fill=(42, 42, 42))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


async def run_benchmark(iterations: int = 30) -> None:
    registry = ModelRegistry()
    width = 640
    height = 480
    image_data = _sample_image_data(width, height)

    print("model_id,runtime,input_size,avg_ms,p95_ms,estimated_fps")
    for model in registry.list_models():
        if not model.available:
            print(f"{model.id},{model.runtime},{model.inputSize},unavailable,{model.unavailableReason},0")
            continue

        detector = registry.detector_for(model.id)
        timings: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            await detector.detect(image_data, width, height, 0.45)
            timings.append((time.perf_counter() - start) * 1000)

        avg_ms = statistics.fmean(timings)
        p95_ms = statistics.quantiles(timings, n=20)[-1] if len(timings) > 1 else avg_ms
        fps = 1000 / avg_ms if avg_ms > 0 else 0
        print(f"{model.id},{model.runtime},{model.inputSize},{avg_ms:.2f},{p95_ms:.2f},{fps:.2f}")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
