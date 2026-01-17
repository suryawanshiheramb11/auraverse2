import sys
import os

# Ensure backend module is found
sys.path.append(os.getcwd())

from backend.processing.temporal import TemporalAggregator
from backend.schemas import Segment

def test_temporal_aggregation():
    print("Testing Temporal Aggregator...")
    
    # Init aggregator at 1 FPS for easy math
    aggregator = TemporalAggregator(fps=1, threshold=0.5)
    
    # Scenario: 
    # Frames 0-1: Real (0.1)
    # Frames 2-4: Fake (0.9)  -> Time 00:00:02 to 00:00:04
    # Frames 5-6: Real (0.1)
    scores = [0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1]
    
    segments = aggregator.aggregate(scores)
    
    if len(segments) != 1:
        print(f"FAILED: Expected 1 segment, got {len(segments)}")
        return False
        
    seg = segments[0]
    # In sliding window logic:
    # Start Frame 2 -> 2.0s
    # End Frame 4 -> 4.0s (This captures the block efficiently)
    
    print(f"Segment Found: {seg.start_time} - {seg.end_time} (Conf: {seg.confidence})")
    
    if seg.start_time == "00:00:02" and seg.end_time == "00:00:04":
        print("PASSED: Timestamps match.")
        return True
    else:
        print(f"FAILED: Timestamps do not match expected 00:00:02-00:00:04")
        return False

if __name__ == "__main__":
    if test_temporal_aggregation():
        print("\nALL PHASE 2 TESTS PASSED")
        sys.exit(0)
    else:
        print("\nPHASE 2 TESTS FAILED")
        sys.exit(1)
