"""OpenVINO Qwen3-ASR worker. The processor and model preparation follow STTMiniTool (MIT)."""
from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
import openvino as ov

from processor_numpy import LightProcessor

SAMPLE_RATE = 16000
WINDOW = 512
MAX_CHUNK = 29 * SAMPLE_RATE


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getframerate() != SAMPLE_RATE or source.getsampwidth() != 2:
            raise ValueError("Expected 16 kHz mono PCM16 WAV")
        return np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def speech_intervals(audio: np.ndarray, model: Path) -> list[tuple[int, int]]:
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    names = {item.name for item in session.get_inputs()}
    state = np.zeros((2, 1, 128), np.float32)
    h = np.zeros((2, 1, 64), np.float32)
    c = np.zeros((2, 1, 64), np.float32)
    context = np.zeros(64, np.float32)
    probabilities = []
    for start in range(0, len(audio), WINDOW):
        chunk = audio[start:start + WINDOW]
        if len(chunk) < WINDOW:
            chunk = np.pad(chunk, (0, WINDOW - len(chunk)))
        if "state" in names:
            output, state = session.run(None, {"input": np.concatenate((context, chunk))[None, :], "state": state, "sr": [SAMPLE_RATE]})
            context = chunk[-64:]
        else:
            output, h, c = session.run(None, {"input": chunk[None, :], "h": h, "c": c, "sr": np.array(SAMPLE_RATE, dtype=np.int64)})
        probabilities.append(float(output[0, 0]))

    raw = []
    start = None
    for index, probability in enumerate(probabilities + [0.0]):
        if probability >= 0.5 and start is None:
            start = index
        elif probability < 0.5 and start is not None:
            if index - start >= 8:
                raw.append((max(0, start - 5) * WINDOW, min(len(audio), (index + 5) * WINDOW)))
            start = None
    merged: list[list[int]] = []
    for begin, end in raw:
        if merged and begin - merged[-1][1] <= 16 * WINDOW:
            merged[-1][1] = end
        else:
            merged.append([begin, end])
    return [(begin, min(len(audio), end + int(0.35 * SAMPLE_RATE))) for begin, end in merged]


def chunks(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result = []
    for begin, end in intervals:
        while begin < end:
            stop = min(begin + MAX_CHUNK, end)
            if stop - begin >= SAMPLE_RATE // 2:
                result.append((begin, stop))
            begin = stop
    return result


class QwenOpenVino:
    def __init__(self, model: Path):
        core = ov.Core()
        config = {"PERFORMANCE_HINT": "LATENCY"}
        self.audio_encoder = core.compile_model(str(model / "audio_encoder_model.xml"), "CPU", config)
        self.embedder = core.compile_model(str(model / "thinker_embeddings_model.xml"), "CPU", config)
        decoder = core.compile_model(str(model / "decoder_model.xml"), "CPU", config)
        self.decoder = decoder.create_infer_request()
        self.processor = LightProcessor(model)

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        mel, ids = self.processor.prepare(audio, language=language)
        encoded = list(self.audio_encoder({"mel": mel}).values())[0]
        combined = list(self.embedder({"input_ids": ids}).values())[0].copy()
        mask = np.where(ids[0] == self.processor.pad_id)[0]
        if len(mask) != encoded.shape[1]:
            raise ValueError("Qwen audio embedding and prompt lengths differ")
        combined[0, mask] = encoded[0]
        length = combined.shape[1]
        self.decoder.reset_state()
        output = self.decoder.infer({0: combined, "position_ids": np.arange(length, dtype=np.int64)[None, :]})
        next_token = int(np.argmax(list(output.values())[0][0, -1]))
        generated = []
        while next_token not in (self.processor.eos_id, self.processor.eot_id) and len(generated) < 300:
            generated.append(next_token)
            embedding = list(self.embedder({"input_ids": np.array([[next_token]], dtype=np.int64)}).values())[0]
            output = self.decoder.infer({0: embedding, "position_ids": np.array([[length]], dtype=np.int64)})
            next_token = int(np.argmax(list(output.values())[0][0, -1]))
            length += 1
        return self.processor.decode(generated).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vad-model", type=Path, required=True)
    parser.add_argument("--vad", choices=("on", "off"), default="on")
    parser.add_argument("--language")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audio = read_wav(args.audio)
    intervals = speech_intervals(audio, args.vad_model) if args.vad == "on" else [(0, len(audio))]
    work = chunks(intervals)
    engine = QwenOpenVino(args.model)
    if args.language and args.language not in engine.processor.supported_languages:
        parser.error(f"Unsupported language: {args.language}")
    result = []
    for begin, end in work:
        text = engine.transcribe(audio[begin:end], args.language)
        if not text:
            continue
        lines = [line.strip() for line in re.split(r"(?<=[.!?。！？])\s*", text) if line.strip()]
        if not lines:
            continue
        duration = (end - begin) / SAMPLE_RATE
        weights = [max(1, len(line)) for line in lines]
        cursor = begin / SAMPLE_RATE
        for index, line in enumerate(lines):
            stop = end / SAMPLE_RATE if index == len(lines) - 1 else cursor + duration * weights[index] / sum(weights)
            result.append({"start": cursor, "end": stop, "text": line})
            cursor = stop
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"OpenVINO worker failed: {error}", file=sys.stderr)
        raise
