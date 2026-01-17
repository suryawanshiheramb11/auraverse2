import random
from typing import List

class MockExpert:
    def predict_frames(self, frame_paths: List[str]) -> List[float]:
        """
        Returns a mock score for each frame.
        Simulating TWO fake segments:
        1. Early in the video (10% - 30%)
        2. Late in the video (70% - 90%)
        """
        scores = []
        total = len(frame_paths)
        for i in range(total):
            # Fake segment 1: 10% to 30%
            # Fake segment 2: 70% to 90%
            if (0.1 * total < i < 0.3 * total) or (0.7 * total < i < 0.9 * total):
                scores.append(random.uniform(0.85, 0.99)) # HIGH CONFIDENCE FAKE
            else:
                scores.append(random.uniform(0.0, 0.15)) # REAL
        return scores

    def predict_image(self, image_path: str) -> float:
        """
        Returns a single mock score for an image.
        Randomly return Fake (0.9) or Real (0.1) for demo.
        """
        # Randomly decide if this image is fake
        return random.choice([0.1, 0.92])
