import cv2
import os
import shutil
from typing import List

class FrameExtractor:
    def __init__(self, output_dir: str = "temp_frames", fps: int = 5):
        self.output_dir = output_dir
        self.target_fps = fps
    
    def extract(self, video_path: str) -> List[str]:
        """
        Extracts frames from video at target_fps.
        Returns list of absolute file paths to the extracted frames.
        """
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / self.target_fps)
        if frame_interval < 1:
            frame_interval = 1

        frame_paths = []
        count = 0
        saved_count = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            if count % frame_interval == 0:
                frame_name = f"frame_{saved_count:05d}.jpg"
                frame_path = os.path.join(self.output_dir, frame_name)
                # Resize for model (standardize to 224x224 later, but keep raw for now)
                # Using 80% quality to save temp space
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_paths.append(os.path.abspath(frame_path))
                saved_count += 1
            
            count += 1
        
        cap.release()
        return frame_paths
