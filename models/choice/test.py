import json
import os
import unittest

from .config import DEBUG_OUTPUT_DIR
from .pipeline import run_pipeline


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.failed = 0
        self.total_images = 0

    def normalize_title(self, s: str) -> str:
        return " ".join(s.lower().strip().split())

    def bbox_close(self, b1, b2, eps: int) -> bool:
        """
        b1, b2: (x_min, y_min, x_max, y_max)
        eps: max allowed pixel deviation per coordinate
        """
        return all(abs(a - b) <= eps for a, b in zip(b1, b2))

    def check_title(
        self,
        title: str,
        expected_title: str,
    ):
        assert title and expected_title is not None
        if self.normalize_title(title) != self.normalize_title(expected_title):
            print(
                f"different titles. expected: {expected_title.lower()}, got: {title.lower()}"
            )
            return False
        return True

    def check_bbox(
        self,
        detected_bbox: tuple,
        expected_bbox: tuple,
        eps: int = 2,
    ):
        """
        tests if detected_bbox is within ε of expected_bbox
        """

        if not self.bbox_close(detected_bbox, expected_bbox, eps):
            print(
                f"Bounding boxes aren't close enough. expected: {expected_bbox}, got: {detected_bbox}"
            )
            return False
        return True

    def test_pipeline(self):
        with open("src/test/input_examples_test/test_bbox.json", "r") as f:
            data = json.load(f)

        with open("src/test/input_examples_test/test_titles.json", "r") as f_titles:
            title_data = json.load(f_titles)

        self.total_images = len(data["images"])

        for card, titles in zip(data["images"], title_data["titles"]):
            annotations = card["annotations"]
            path = card["image"]
            path = os.path.join("src", "test", "img_input_test", path)
            try:
                pairs = run_pipeline(path)
            except Exception as e:
                with open(
                    os.path.join(DEBUG_OUTPUT_DIR, os.path.basename(path)), "w"
                ) as debug_file:
                    msg = f"pipeline failed with exception: {e} for file {os.path.basename(path)}"
                    debug_file.write(msg)
                    print(msg)
                    continue

            expected_results = []
            for annot, title in zip(annotations, titles):
                ann = annot["boundingBox"]
                expected_bbox = (
                    ann["x"],
                    ann["y"],
                    ann["width"],
                    ann["height"],
                )
                expected_results.append((expected_bbox, title))

            for e, pred in zip(expected_results, pairs):
                print(f"expected: {e}")
                print(f"predicted: {pred}")
                t = pred["title"]
                exp_t = e[1]
                exp_bbox = e[0]
                bbox = pred["det_xywh"]
                if not self.check_bbox(bbox, exp_bbox, 50) or self.check_title(
                    t, exp_t
                ):
                    self.failed += 1
                    break

        success = self.total_images - self.failed
        print(
            f"{success} / {self.total_images} successful. Accuracy: {success / self.total_images}"
        )


if __name__ == "__main__":
    unittest.main()
