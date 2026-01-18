import cv2
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IngestVideo")

VIDEO_PATH = "/Users/suryawanshiheramb11/Downloads/ai videos/Movie on 18-01-26 at 12.46 AM.mov"
OUTPUT_DIR = "dataset/real_user_submission"

def extract_frames():
    if not os.path.exists(VIDEO_PATH):
        logger.error(f"Video not found at {VIDEO_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        logger.error("Could not open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video Info: {total_frames} frames, {fps} FPS")

    # Target: Extract ~150 frames
    # If video is short, extract all. If long, skip.
    TARGET_COUNT = 150
    stride = max(1, total_frames // TARGET_COUNT)
    
    count = 0
    saved = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % stride == 0:
            # Save frame
            filename = f"frame_{count:05d}.jpg"
            out_path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(out_path, frame)
            saved += 1
            if saved % 50 == 0:
                logger.info(f"Saved {saved} frames...")
                
        count += 1

    cap.release()
    logger.info(f"Ingestion Complete. Saved {saved} frames to {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_frames()
