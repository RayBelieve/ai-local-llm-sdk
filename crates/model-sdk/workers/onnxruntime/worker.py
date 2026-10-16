"""Object detection for the pinned EdgeFirst YOLOv8/YOLO11 ONNX exports."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


INPUT_SIZE = 640


def prepare_image(path: Path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    width, height = image.size
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    resized = (round(width * scale), round(height * scale))
    left = (INPUT_SIZE - resized[0]) // 2
    top = (INPUT_SIZE - resized[1]) // 2
    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (114, 114, 114))
    canvas.paste(image.resize(resized, Image.Resampling.BILINEAR), (left, top))
    tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return tensor, width, height, scale, left, top


def iou(box, others):
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    overlap = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area = (box[2] - box[0]) * (box[3] - box[1])
    other_area = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return overlap / np.maximum(area + other_area - overlap, 1e-9)


def decode(output, labels, width, height, scale, left, top, confidence, iou_threshold, maximum):
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != len(labels) + 4:
        raise ValueError(f"Unsupported YOLO output shape: {output.shape}")
    rows = output[0].T
    class_ids = np.argmax(rows[:, 4:], axis=1)
    scores = rows[np.arange(len(rows)), class_ids + 4]
    keep = np.flatnonzero(np.isfinite(scores) & (scores >= confidence))
    if not len(keep):
        return []
    centers = rows[keep, :4] * INPUT_SIZE
    boxes = np.column_stack((
        (centers[:, 0] - centers[:, 2] / 2 - left) / scale,
        (centers[:, 1] - centers[:, 3] / 2 - top) / scale,
        (centers[:, 0] + centers[:, 2] / 2 - left) / scale,
        (centers[:, 1] + centers[:, 3] / 2 - top) / scale,
    ))
    boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, width)
    boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, height)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores, class_ids = boxes[valid], scores[keep][valid], class_ids[keep][valid]
    order = np.argsort(-scores)
    detections = []
    while len(order) and len(detections) < maximum:
        index = order[0]
        box = boxes[index]
        class_id = int(class_ids[index])
        detections.append({
            "classId": class_id,
            "label": labels[class_id],
            "confidence": float(scores[index]),
            "x1": float(box[0]), "y1": float(box[1]),
            "x2": float(box[2]), "y2": float(box[3]),
        })
        remaining = order[1:]
        same_class = class_ids[remaining] == class_id
        suppress = np.zeros(len(remaining), dtype=bool)
        suppress[same_class] = iou(box, boxes[remaining[same_class]]) > iou_threshold
        order = remaining[~suppress]
    return detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--confidence", required=True, type=float)
    parser.add_argument("--iou", required=True, type=float)
    parser.add_argument("--max-detections", required=True, type=int)
    args = parser.parse_args()

    import onnxruntime as ort

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    labels = json.loads(session.get_modelmeta().custom_metadata_map["labels"])
    if len(labels) != 80 or session.get_inputs()[0].shape != [1, 3, INPUT_SIZE, INPUT_SIZE]:
        raise ValueError("Unsupported YOLO model metadata")
    tensor, width, height, scale, left, top = prepare_image(args.image)
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    detections = decode(
        output, labels, width, height, scale, left, top,
        args.confidence, args.iou, args.max_detections,
    )
    args.output.write_text(json.dumps({
        "imageWidth": width, "imageHeight": height, "detections": detections,
    }), encoding="utf-8")


if __name__ == "__main__":
    main()
