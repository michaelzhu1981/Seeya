import numpy as np

from app.detectors.onnx_detector import LetterboxInfo, _nms, _postprocess_yolov8


def test_postprocess_yolov8_maps_boxes_to_source_coordinates() -> None:
    output = np.zeros((1, 84, 2), dtype=np.float32)
    output[0, 0:4, 0] = [208, 208, 104, 156]
    output[0, 4, 0] = 0.92
    output[0, 0:4, 1] = [100, 100, 20, 20]
    output[0, 4, 1] = 0.2

    detections = _postprocess_yolov8(
        output,
        LetterboxInfo(
            ratio=0.65,
            pad_x=0,
            pad_y=52,
            source_width=640,
            source_height=480,
            input_size=416,
        ),
        confidence_threshold=0.45,
        iou_threshold=0.45,
        max_detections=100,
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection.label == "person"
    assert detection.confidence == float(np.float32(0.92))
    assert round(detection.box.x) == 240
    assert round(detection.box.y) == 120
    assert round(detection.box.width) == 160
    assert round(detection.box.height) == 240


def test_nms_keeps_highest_scoring_overlapping_box() -> None:
    boxes = np.array(
        [
            [10, 10, 110, 110],
            [15, 15, 105, 105],
            [200, 200, 240, 240],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.7, 0.9, 0.5], dtype=np.float32)

    assert _nms(boxes, scores, iou_threshold=0.45) == [1, 2]
