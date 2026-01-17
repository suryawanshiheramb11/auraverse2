import torch
import os
import logging
from model_core import SentinelHybrid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelVerifier")

def verify():
    MODEL_PATH = "sentinel_model.pth"
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Model file not found at: {MODEL_PATH}")
        return
        
    logger.info(f"✅ Found model file: {MODEL_PATH}")
    
    try:
        # Initialize Model
        model = SentinelHybrid()
        logger.info("Initialized SentinelHybrid architecture.")
        
        # Load State Dict
        logger.info("Attempting to load state dictionary...")
        state_dict = torch.load(MODEL_PATH, map_location=model.device)
        model.load_state_dict(state_dict)
        
        logger.info("✅ SUCCESS: Trained weights loaded successfully!")
        logger.info("The architecture matches the checkpoint.")
        
    except RuntimeError as e:
        logger.error(f"❌ LOADING FAILED: Size Mismatch or Architecture Mismatch.")
        logger.error(f"Error Details: {e}")
        logger.info("Hint: Did you change the model architecture (e.g. hidden size, layers) in `model_core.py` after training?")
        
    except Exception as e:
        logger.error(f"❌ LOADING FAILED: {e}")

if __name__ == "__main__":
    verify()
