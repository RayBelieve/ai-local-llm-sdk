import unittest

import numpy as np

from worker import decode, iou


class DetectionTests(unittest.TestCase):
    def test_class_aware_nms_and_letterbox_coordinates(self):
        output = np.zeros((1, 84, 3), dtype=np.float32)
        output[0, :4, :] = np.array([
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25],
        ])
        output[0, 4, :2] = [0.9, 0.8]
        output[0, 5, 2] = 0.7
        labels = [str(index) for index in range(80)]
        result = decode(output, labels, 1280, 640, 0.5, 0, 160, 0.25, 0.45, 100)
        self.assertEqual([item["classId"] for item in result], [0, 1])
        self.assertEqual([result[0][key] for key in ("x1", "y1", "x2", "y2")], [480, 160, 800, 480])

    def test_iou_handles_nonoverlapping_boxes(self):
        value = iou(np.array([0, 0, 10, 10]), np.array([[20, 20, 30, 30]]))
        self.assertEqual(value.tolist(), [0.0])


if __name__ == "__main__":
    unittest.main()
