import unittest

import numpy as np
from PIL import Image

from tag_worker import decode, prepare_image


class TagWorkerTests(unittest.TestCase):
    def test_returns_all_scores_and_categories(self):
        rows = [{"name": name, "category": category} for name, category in [
            ("general", "9"), ("sensitive", "9"), ("questionable", "9"),
            ("explicit", "9"), ("1girl", "0"), ("alice", "4")]]
        result = decode(rows, np.array([[0.8, 0.1, 0.05, 0.05, 0.99, 0.001]]))
        self.assertEqual(len(result["ratings"]), 4)
        self.assertEqual(result["tags"][-1], {"name": "alice", "category": 4, "score": 0.001})

    def test_alpha_is_white_and_input_is_bgr(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGBA", (1, 1), (255, 0, 0, 0)).save(path)
            tensor = prepare_image(path, 2)
            self.assertEqual(tensor.shape, (1, 2, 2, 3))
            self.assertEqual(tensor[0, 0, 0].tolist(), [255.0, 255.0, 255.0])


if __name__ == "__main__":
    unittest.main()
