from typing import List, Dict, Optional
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import logging
import math
import sys

# Add root logic to path if needed for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.processing.frames import FrameExtractor
from backend.schemas import DeepfakeResponse, Segment, Evidence
from model_core import SentinelHybrid

# Setup Logging
logger = logging.getLogger("Orchestrator")

class Orchestrator:
    def __init__(self):
        self.frame_extractor = FrameExtractor(fps=2) # Optimized for speed
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Orchestrator initialized on device: {self.device}")
        
        self.model = self._load_model()
        self.transform = self._get_transforms()

    def _load_model(self):
        """Loads the SentinelHybrid model with trained weights."""
        try:
            model = SentinelHybrid(sequence_length=10)
            
            # Path to the model file - assuming it's in the root folder
            # Adjust path resolution as needed
            input_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "sentinel_model.pth")
            
            if not os.path.exists(input_model_path):
                 # Fallback for hackathon: try absolute path or current dir
                 input_model_path = "sentinel_model.pth"
            
            if os.path.exists(input_model_path):
                logger.info(f"Loading weights from {input_model_path}")
                state_dict = torch.load(input_model_path, map_location=self.device)
                model.load_state_dict(state_dict)
            else:
                logger.error("CRITICAL: sentinel_model.pth not found! Using random weights.")
            
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def _get_transforms(self):
        """Returns the preprocessing transforms required by EfficientNet."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _format_time(self, seconds: float) -> str:
        """Formats seconds into MM:SS string."""
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def process_video(self, video_path: str) -> DeepfakeResponse:
        """
        Full pipeline: Video -> Frames -> Batching (Seq 10) -> Inference -> Response
        """
        logger.info(f"Processing video: {video_path}")
        
        # 1. Extract Frames
        frame_paths = self.frame_extractor.extract(video_path)
        if not frame_paths:
            logger.warning("No frames extracted.")
            return DeepfakeResponse(input_type="video", video_is_fake=False, overall_confidence=0.0, fake_score=0.0, manipulated_segments=[], evidence=[])

        # 2. Preprocess & Batch
        # We need sequences of 10 frames
        SEQUENCE_LENGTH = 10
        timestamps = [] # To track time mappings
        
        # Sort frames to ensure temporal order
        frame_paths.sort()
        
        current_seq = []
        
        manipulated_segments = []
        max_conf = 0.0
        evidence = []
        fake_frame_count = 0
        total_frames = len(frame_paths)
        
        # Ensure output dir for evidence exists
        evidence_dir = "static/evidence"
        os.makedirs(evidence_dir, exist_ok=True)
        
        with torch.no_grad():
            for i, fpath in enumerate(frame_paths):
                try:
                    img = Image.open(fpath).convert('RGB')
                    tensor = self.transform(img)
                    current_seq.append(tensor)
                    
                    if len(current_seq) == SEQUENCE_LENGTH:
                        # Stack
                        input_tensor = torch.stack(current_seq).unsqueeze(0) # (1, 10, 3, 224, 224)
                        input_tensor = input_tensor.to(self.device)
                        
                        # Inference
                        logits = self.model(input_tensor) # (1, 2)
                        
                        # Softmax
                        probs = torch.softmax(logits, dim=1)
                        fake_prob = probs[0][1].item() # Class 1 is Fake
                        
                        # NaN Safe
                        if math.isnan(fake_prob):
                            fake_prob = 0.0
                        
                        # Update stats
                        if fake_prob > max_conf:
                            max_conf = fake_prob
                        
                        # Create segment timestamps
                        start_sec = (i - 9) / 5.0
                        end_sec = i / 5.0
                        start_time = self._format_time(start_sec)
                        end_time = self._format_time(end_sec)
                        
                        # Register meaningful segments (threshold e.g., 0.5)
                        if fake_prob > 0.5:
                             fake_frame_count += SEQUENCE_LENGTH # Rough estimate
                             manipulated_segments.append(Segment(
                                 start_time=start_time,
                                 end_time=end_time,
                                 confidence=fake_prob
                             ))
                             
                             # Save evidence if it's a strong positive
                             if fake_prob > 0.7:
                                 # Save the middle frame of the sequence as evidence
                                 import shutil
                                 import time
                                 
                                 evidence_name = f"evidence_{int(time.time()*1000)}_{i}.jpg"
                                 evidence_dest = os.path.join(evidence_dir, evidence_name)
                                 shutil.copy(fpath, evidence_dest)
                                 
                                 # Calculate timestamp for seeking (middle of sequence)
                                 seek_timestamp = max(0.0, start_sec + (end_sec - start_sec) / 2)
                                 
                                 evidence.append(Evidence(
                                     path=f"/static/evidence/{evidence_name}",
                                     timestamp=seek_timestamp,
                                     frame_index=i
                                 ))
                        
                        # Reset
                        current_seq = []
                except Exception as e:
                    logger.error(f"Error processing frame {fpath}: {e}")
                    continue

        is_fake = max_conf > 0.5
        
        # Calculate fake score (extent of fakeness)
        fake_score = 0.0
        if total_frames > 0:
             fake_score = min(100.0, (fake_frame_count / total_frames) * 100)
             
        if math.isnan(fake_score): fake_score = 0.0
        if math.isnan(max_conf): max_conf = 0.0
        
        logger.info(f"Video Analysis Complete. Fake: {is_fake}, Conf: {max_conf}, Score: {fake_score}")
        
        # Limit evidence (deduplicate logic could be added but simpler is fine)
        # We need to deduplicate based on path or roughly same timestamp to avoid spam
        # But for now, just slice
        evidence = evidence[:12]
        
        return DeepfakeResponse(
            input_type="video",
            video_is_fake=is_fake,
            overall_confidence=max_conf,
            fake_score=fake_score,
            manipulated_segments=manipulated_segments,
            evidence=evidence
        )

    def process_image(self, image_path: str) -> DeepfakeResponse:
        """
        Inference for a single image.
        We can duplicate the image 10 times to satisfy the sequence requirement of the model.
        """
        logger.info(f"Processing image: {image_path}")
        try:
             with torch.no_grad():
                 img = Image.open(image_path).convert('RGB')
                 tensor = self.transform(img)
                 
                 # Create a sequence of 10 identical frames
                 seq = torch.stack([tensor] * 10).unsqueeze(0) # (1, 10, 3, 224, 224)
                 seq = seq.to(self.device)
                 
                 logits = self.model(seq)
                 probs = torch.softmax(logits, dim=1)
                 fake_prob = probs[0][1].item()
                 
                 if math.isnan(fake_prob): fake_prob = 0.0
                 
                 is_fake = fake_prob > 0.5
                 
                 return DeepfakeResponse(
                    input_type="image",
                    video_is_fake=is_fake,
                    overall_confidence=fake_prob,
                    fake_score=fake_prob * 100 if is_fake else 0.0,
                    manipulated_segments=[],
                    evidence=[]
                )
        except Exception as e:
             logger.error(f"Image analysis failed: {e}")
             return DeepfakeResponse(input_type="image", video_is_fake=False, overall_confidence=0.0, fake_score=0.0, manipulated_segments=[], evidence=[])
