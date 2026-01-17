import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestOrchestrator")

# Add root to path
sys.path.append(os.getcwd())

try:
    from backend.orchestrator import Orchestrator
    logger.info("Attempting to initialize Orchestrator...")
    orchestrator = Orchestrator()
    logger.info("✅ Orchestrator initialized successfully!")
    logger.info(f"Model Device: {orchestrator.device}")
except Exception as e:
    logger.error(f"❌ Initialization Failed: {e}")
    exit(1)
