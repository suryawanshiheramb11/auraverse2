import cv2
import os
import shutil
from typing import List

class FrameExtractor:
    def __init__(self, output_dir: str = "temp_frames", fps: int = 5):
        self.output_dir = output_dir
        self.target_fps = fps
    
    def extract(self, video_path: str, max_frames: int = 300) -> List[str]:
        """
        Extracts frames from video.
        - If video is short: Extracts at target_fps.
        - If video is long: Smartly samples 'max_frames' distributed across the duration.
        Returns list of absolute file paths.
        """
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_video_frames / video_fps if video_fps > 0 else 0
        
        # Calculate sampling interval
        # We want at most 'max_frames'
        # Nominal interval for target_fps
        nominal_interval = int(video_fps / self.target_fps) if self.target_fps > 0 else 1
        if nominal_interval < 1: nominal_interval = 1
        
        # Check if nominal extraction would exceed max_frames
        expected_frames = total_video_frames // nominal_interval
        
        if expected_frames > max_frames:
            # We need to increase the interval to stay under max_frames
            frame_interval = int(total_video_frames / max_frames)
        else:
            frame_interval = nominal_interval
            
        if frame_interval < 1: frame_interval = 1

        frame_paths = []
        saved_count = 0
        current_frame = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                break

            if current_frame % frame_interval == 0:
                # Limit safety
                if saved_count >= max_frames:
                    break
                    
                frame_name = f"frame_{saved_count:05d}.jpg"
                frame_path = os.path.join(self.output_dir, frame_name)
                
                # Resize to reduce I/O time (Smart resize)
                # If image is huge (4k), resize to 640px max dim for speed
                h, w = frame.shape[:2]
                if h > 640 or w > 640:
                    scale = 640 / max(h, w)
                    new_size = (int(w * scale), int(h * scale))
                    frame = cv2.resize(frame, new_size)

                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_paths.append(os.path.abspath(frame_path))
                saved_count += 1
            
            current_frame += 1
        
        cap.release()
        return frame_paths
