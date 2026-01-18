from typing import List, Dict, Optional
import os
import sys
import logging
from backend.schemas import DeepfakeResponse, Segment, Evidence

# Add root logic to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local forensics engine (Uses Torch + OpenCV, NO MediaPipe/Transformers)
import forensics_engine

logger = logging.getLogger("Orchestrator")

class Orchestrator:
    def __init__(self):
        logger.info(f"Orchestrator initialized (Legacy Engine Mode).")
        # No heavy init needed effectively, as forensics_engine initializes globally or on demand

    def process_video(self, video_path: str) -> DeepfakeResponse:
        try:
            # Delegate to forensics_engine
            # It returns a dict with: input_type, video_is_fake, fake_score, manipulated_segments, evidence_paths
            logger.info(f"Delegating video analysis to ForensicsEngine: {video_path}")
            result = forensics_engine.analyze_video_segments(video_path)
            
            # Map result to DeepfakeResponse
            segments = []
            for seg in result.get("manipulated_segments", []):
                # seg has start (str), end (str), confidence (float)
                segments.append(Segment(
                    start_time=seg["start"],
                    end_time=seg["end"],
                    confidence=seg["confidence"]
                ))
            
            evidence_list = []
            for path in result.get("evidence_paths", []):
                # Parse frame index from path if possible: evidence_TIMESTAMP_FRAME.jpg
                frame_idx = 0
                ts = 0.0
                try:
                    basename = os.path.basename(path)
                    parts = basename.split('_')
                    # format: evidence_{timestamp_ms}_{frame_idx}.jpg
                    if len(parts) >= 3:
                        frame_idx = int(parts[-1].split('.')[0])
                        # Estimate timestamp: frame / 30 (default fps assumption)
                        ts = frame_idx / 30.0 
                except:
                    pass
                
                evidence_list.append(Evidence(
                    path=path,
                    timestamp=ts,
                    frame_index=frame_idx
                ))

            # fake_score in result is 0-100
            score = result.get("fake_score", 0.0)
            
            response = DeepfakeResponse(
                input_type="video",
                video_is_fake=result.get("video_is_fake", False),
                overall_confidence=score / 100.0,
                fake_score=score,
                manipulated_segments=segments,
                evidence=evidence_list
            )
            return response
            
        except Exception as e:
            logger.error(f"Analysis Failed: {e}")
            import traceback
            traceback.print_exc()
            # Return safe error response
            return DeepfakeResponse(
                input_type="video", 
                video_is_fake=False, 
                overall_confidence=0.0, 
                fake_score=0.0,
                manipulated_segments=[],
                evidence=[]
            )

    def process_image(self, image_path: str) -> DeepfakeResponse:
         try:
            logger.info(f"Delegating image analysis to ForensicsEngine: {image_path}")
            result = forensics_engine.analyze_image(image_path)
            # Result keys: input_type, video_is_fake, confidence, manipulated_segments
            
            # confidence is 0.0-1.0
            conf = result.get("confidence", 0.0)
            fake_score = conf * 100
            
            return DeepfakeResponse(
                input_type="image",
                video_is_fake=result.get("video_is_fake", False),
                overall_confidence=conf,
                fake_score=fake_score,
                manipulated_segments=[],
                evidence=[],
                preview_url=result.get("preview_url")
            )
         except Exception as e:
            logger.error(f"Image Analysis Failed: {e}")
            import traceback
            traceback.print_exc()
            return DeepfakeResponse(input_type="image", video_is_fake=False, overall_confidence=0.0, fake_score=0.0, manipulated_segments=[], evidence=[])
