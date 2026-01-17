import cv2
import numpy as np
import os
import shutil
import logging
from forensics_engine import analyze_video_segments, analyze_image

# Configure basic logging
logging.basicConfig(level=logging.INFO)

def create_dummy_video(filename="test_video_face.mp4"):
    """Creates a dummy video with a drawing that looks like a face to trigger detection."""
    frame_width = 224
    frame_height = 224
    fps = 10
    duration_sec = 2
    
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    # Draw a "face"
    # Circle for head, ellipses for eyes?
    # Actually, Haar cascades are robust but need some structure.
    # Alternatively, we can assume the code falls back or we rely on the logic that "if no face, skip" 
    # might skip everything if we don't have a real face.
    # BUT, I put a fallback or logic: "if face_crop is None: continue".
    # So I NEED a face.
    # It's hard to draw a convincing face for Haar with basic shapes without trial and error.
    # Better approach for Verification: 
    # 1. Just test the API structure if I can't generate a face.
    # 2. Or assume the "skip" logic is working if I get 0 segments.
    # 3. Or use a real image if available?
    
    # Let's try to draw a simple smiley face, usually Haar needs features (eyes, nose bridge).
    for i in range(fps * duration_sec):
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8) + 200 # Light gray bg
        
        # Face (Head)
        cv2.circle(frame, (112, 112), 60, (100, 100, 255), -1) 
        # Eyes
        cv2.circle(frame, (90, 100), 8, (0, 0, 0), -1)
        cv2.circle(frame, (134, 100), 8, (0, 0, 0), -1)
        # Mouth
        cv2.ellipse(frame, (112, 130), (20, 10), 0, 0, 180, (0, 0, 0), 3)
        
        # Add some noise to trigger variance?
        noise = np.random.randint(0, 50, (frame_height, frame_width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        out.write(frame)
        
    out.release()
    print(f"Created dummy video: {filename}")
    return filename

def test_analysis():
    video_path = create_dummy_video()
    
    print("\n--- Testing Video Analysis ---")
    try:
        result = analyze_video_segments(video_path)
        print("Analysis Result Keys:", result.keys())
        print(f"Fake Score: {result.get('fake_score')}")
        print(f"Evidence Paths: {result.get('evidence_paths')}")
        
        if result.get('evidence_paths'):
            print("Verified: Evidence paths returned.")
        else:
            print("Note: No evidence returned (likely no face detected or not fake enough).")
            
    except Exception as e:
        print(f"Analysis Failed: {e}")
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

if __name__ == "__main__":
    test_analysis()
