import argparse
import json
import re
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
from misaki.zh import ZHG2P

SAMPLE_RATE = 24000
MAX_PHONEMES = 510


def phoneme_chunks(text, g2p):
    for paragraph in text.splitlines():
        for sentence in re.split(r"(?<=[。！？.!?；;])", paragraph):
            sentence = sentence.strip()
            for start in range(0, len(sentence), 100):
                piece = sentence[start : start + 100]
                phonemes, _ = g2p(piece)
                if not phonemes or phonemes == "❓":
                    raise ValueError(f"文本无法转换为中文音素: {piece}")
                if len(phonemes) > MAX_PHONEMES:
                    raise ValueError("合成文本片段过长，请缩短句子。")
                yield phonemes


def synthesize(model, text, speed):
    vocab = json.loads((model / "tokenizer.json").read_text(encoding="utf-8"))["model"]["vocab"]
    voices = np.fromfile(model / "zf_001.bin", dtype=np.float32).reshape(510, 256)
    session = ort.InferenceSession(str(model / "model.onnx"), providers=["CPUExecutionProvider"])
    g2p = ZHG2P(version="1.1")
    clips = []
    for phonemes in phoneme_chunks(text, g2p):
        try:
            ids = [0, *(vocab[char] for char in phonemes), 0]
        except KeyError as error:
            raise ValueError(f"模型词表不支持音素: {error}") from error
        audio = session.run(None, {
            "input_ids": np.asarray([ids], dtype=np.int64),
            "style": voices[len(phonemes) - 1 : len(phonemes)],
            "speed": np.asarray([speed], dtype=np.float32),
        })[0]
        clips.append(np.asarray(audio, dtype=np.float32).reshape(-1))
    if not clips:
        raise ValueError("合成文本不能为空。")
    return np.concatenate([item for clip in clips for item in (clip, np.zeros(2400, dtype=np.float32))][:-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed", type=float, required=True)
    args = parser.parse_args()
    audio = synthesize(args.model, args.text_file.read_text(encoding="utf-8"), args.speed)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(args.output), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm.tobytes())


if __name__ == "__main__":
    main()
