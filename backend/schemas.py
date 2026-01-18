from pydantic import BaseModel
from typing import List, Literal, Optional

class Segment(BaseModel):
    start_time: str
    end_time: str
    confidence: float

class Evidence(BaseModel):
    path: str
    timestamp: float # Seconds
    frame_index: int

class DeepfakeResponse(BaseModel):
    input_type: Literal["video", "image"]
    video_is_fake: bool
    overall_confidence: float
    fake_score: float = 0.0
    manipulated_segments: List[Segment]
    evidence: List[Evidence] = []
    preview_url: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
