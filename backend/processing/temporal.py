from typing import List
from backend.schemas import Segment

class TemporalAggregator:
    def __init__(self, fps: int = 5, threshold: float = 0.5):
        self.fps = fps
        self.threshold = threshold

    def aggregate(self, frame_scores: List[float]) -> List[Segment]:
        """
        Converts a list of frame-level probabilities (e.g. [0.1, 0.9, 0.9])
        into temporal segments (Start=00:00:01, End=00:00:05).
        """
        segments = []
        in_segment = False
        start_frame_idx = 0
        
        # Add a dummy safe frame at the end to close any open segments
        extended_scores = frame_scores + [0.0]

        for i, score in enumerate(extended_scores):
            is_fake = score > self.threshold
            
            if is_fake and not in_segment:
                # Start of a new fake segment
                in_segment = True
                start_frame_idx = i
            
            elif not is_fake and in_segment:
                # End of a fake segment
                in_segment = False
                end_frame_idx = i - 1 # Previous frame was the last fake one
                
                # Convert frame indices to timestamps
                start_time = self._frame_to_timestamp(start_frame_idx)
                end_time = self._frame_to_timestamp(end_frame_idx)
                
                # Calculate average confidence for this segment
                segment_scores = frame_scores[start_frame_idx : end_frame_idx + 1]
                avg_conf = sum(segment_scores) / len(segment_scores) if segment_scores else 0.0
                
                segments.append(Segment(
                    start_time=start_time,
                    end_time=end_time,
                    confidence=round(avg_conf, 2)
                ))

        return segments

    def _frame_to_timestamp(self, frame_idx: int) -> str:
        total_seconds = frame_idx / self.fps
        mins = int(total_seconds // 60)
        secs = int(total_seconds % 60)
        return f"00:{mins:02d}:{secs:02d}"
