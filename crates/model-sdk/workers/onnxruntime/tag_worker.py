"""Raw WD-14 scores for a pinned SmilingWolf model and tag list."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def prepare_image(path, size):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
    background = Image.new("RGBA", image.size, "white")
    background.alpha_composite(image)
    image = background.convert("RGB")
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    image = canvas.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32)[:, :, ::-1].copy()[None]


def decode(rows, output):
    scores = np.asarray(output)
    if scores.shape != (1, len(rows)) or not np.isfinite(scores).all():
        raise ValueError(f"Unsupported WD-14 output shape or scores: {scores.shape}")
    entries = [{"name": row["name"], "category": int(row["category"]), "score": float(score)}
               for row, score in zip(rows, scores[0])]
    ratings = [entry for entry in entries if entry["category"] == 9]
    if len(ratings) != 4:
        raise ValueError("WD-14 tag list must contain four ratings")
    return {"ratings": ratings, "tags": [entry for entry in entries if entry["category"] != 9]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import onnxruntime as ort

    with (args.model / "selected_tags.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    session = ort.InferenceSession(str(args.model / "model.onnx"), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    shape = input_info.shape
    if len(shape) != 4 or shape[1] != shape[2] or shape[3] != 3 or not isinstance(shape[1], int):
        raise ValueError(f"Unsupported WD-14 input shape: {shape}")
    tensor = prepare_image(args.image, shape[1])
    result = decode(rows, session.run(None, {input_info.name: tensor})[0])
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
