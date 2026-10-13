import unittest
from unittest.mock import patch

import numpy as np

from worker import SAMPLE_RATE, WINDOW, chunks, speech_intervals


class FakeSession:
    def __init__(self, *_args, **_kwargs):
        self.index = 0

    def get_inputs(self):
        return [type("Input", (), {"name": name})() for name in ("input", "h", "c", "sr")]

    def run(self, _outputs, inputs):
        active = 30 <= self.index < 65
        self.index += 1
        return np.array([[0.8 if active else 0.0]], np.float32), inputs["h"], inputs["c"]


class WorkerTests(unittest.TestCase):
    def test_vad_preserves_original_offset(self):
        audio = np.zeros(SAMPLE_RATE * 3, np.float32)
        with patch("worker.ort.InferenceSession", FakeSession):
            intervals = speech_intervals(audio, "unused")
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0][0], 25 * WINDOW)
        self.assertGreater(intervals[0][1], 65 * WINDOW)

    def test_long_interval_is_split_without_losing_offset(self):
        intervals = chunks([(SAMPLE_RATE * 10, SAMPLE_RATE * 75)])
        self.assertEqual(intervals[0][0], SAMPLE_RATE * 10)
        self.assertEqual(intervals[-1][1], SAMPLE_RATE * 75)
        self.assertTrue(all(end - start <= SAMPLE_RATE * 29 for start, end in intervals))


if __name__ == "__main__":
    unittest.main()
