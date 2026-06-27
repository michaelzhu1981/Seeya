from __future__ import annotations

import asyncio
import base64
import binascii
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.detectors.base import Detector
from app.schemas import Box, Detection


COCO_LABELS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


@dataclass(frozen=True)
class LetterboxInfo:
    ratio: float
    pad_x: float
    pad_y: float
    source_width: int
    source_height: int
    input_size: int


class OnnxDetector(Detector):
    model_id = "yolo-nano-onnx-cpu"

    def __init__(
        self,
        model_path: Path,
        *,
        model_id: str = "yolo-nano-onnx-cpu",
        providers: list[str] | None = None,
        input_size: int = 416,
        max_detections: int = 100,
        iou_threshold: float = 0.45,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is not installed") from exc

        self.model_id = model_id
        self.model_path = model_path
        self.input_size = input_size
        self.max_detections = max_detections
        self.iou_threshold = iou_threshold
        self.session = ort.InferenceSession(str(model_path), providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    async def detect(
        self,
        image_data: str,
        width: int,
        height: int,
        confidence_threshold: float,
    ) -> list[Detection]:
        image = _decode_image(image_data)
        tensor, letterbox = _preprocess_image(image, self.input_size, width, height)
        output = await asyncio.to_thread(self._run_session, tensor)
        return _postprocess_yolov8(
            output,
            letterbox,
            confidence_threshold=confidence_threshold,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections,
        )

    def _run_session(self, tensor: np.ndarray) -> np.ndarray:
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        return outputs[0]


def _decode_image(image_data: str) -> Image.Image:
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    try:
        raw = base64.b64decode(image_data, validate=True)
    except binascii.Error as exc:
        raise ValueError("imageData must be a base64 encoded image") from exc
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _preprocess_image(
    image: Image.Image,
    input_size: int,
    source_width: int,
    source_height: int,
) -> tuple[np.ndarray, LetterboxInfo]:
    image = image.resize((source_width, source_height)) if image.size != (source_width, source_height) else image
    ratio = min(input_size / source_width, input_size / source_height)
    resized_width = int(round(source_width * ratio))
    resized_height = int(round(source_height * ratio))
    pad_x = (input_size - resized_width) / 2
    pad_y = (input_size - resized_height) / 2

    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (input_size, input_size), (114, 114, 114))
    canvas.paste(resized, (int(round(pad_x)), int(round(pad_y))))

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = np.transpose(array, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(tensor), LetterboxInfo(
        ratio=ratio,
        pad_x=pad_x,
        pad_y=pad_y,
        source_width=source_width,
        source_height=source_height,
        input_size=input_size,
    )


def _postprocess_yolov8(
    output: np.ndarray,
    letterbox: LetterboxInfo,
    *,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> list[Detection]:
    predictions = np.squeeze(output)
    if predictions.ndim != 2:
        raise ValueError(f"Unexpected YOLO output shape: {output.shape}")
    if predictions.shape[0] in {len(COCO_LABELS) + 4, len(COCO_LABELS) + 5}:
        predictions = predictions.T

    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 5:] if predictions.shape[1] == len(COCO_LABELS) + 5 else predictions[:, 4:]
    if class_scores.size == 0:
        return []

    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
    keep = confidences >= confidence_threshold
    if not np.any(keep):
        return []

    boxes = _xywh_to_xyxy(boxes_xywh[keep])
    boxes = _scale_boxes_to_source(boxes, letterbox)
    confidences = confidences[keep]
    class_ids = class_ids[keep]

    nms_indices = _nms(boxes, confidences, iou_threshold)[:max_detections]
    detections: list[Detection] = []
    for index in nms_indices:
        x1, y1, x2, y2 = boxes[index]
        label_index = int(class_ids[index])
        label = COCO_LABELS[label_index] if label_index < len(COCO_LABELS) else f"class {label_index}"
        detections.append(
            Detection(
                label=label,
                confidence=float(confidences[index]),
                box=Box(
                    x=float(x1),
                    y=float(y1),
                    width=float(max(0.0, x2 - x1)),
                    height=float(max(0.0, y2 - y1)),
                ),
            )
        )
    return detections


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = np.empty_like(boxes, dtype=np.float32)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def _scale_boxes_to_source(boxes: np.ndarray, letterbox: LetterboxInfo) -> np.ndarray:
    scaled = boxes.copy()
    scaled[:, [0, 2]] = (scaled[:, [0, 2]] - letterbox.pad_x) / letterbox.ratio
    scaled[:, [1, 3]] = (scaled[:, [1, 3]] - letterbox.pad_y) / letterbox.ratio
    scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0, letterbox.source_width)
    scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0, letterbox.source_height)
    return scaled


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        remaining = order[1:]
        xx1 = np.maximum(x1[current], x1[remaining])
        yy1 = np.maximum(y1[current], y1[remaining])
        xx2 = np.minimum(x2[current], x2[remaining])
        yy2 = np.minimum(y2[current], y2[remaining])

        widths = np.maximum(0.0, xx2 - xx1)
        heights = np.maximum(0.0, yy2 - yy1)
        intersection = widths * heights
        union = areas[current] + areas[remaining] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = remaining[iou <= iou_threshold]

    return keep
