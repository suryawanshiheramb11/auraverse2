from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import os
import tempfile
import logging
import sys

# Add root logic to path if needed for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.orchestrator import Orchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SentinelServer")

app = FastAPI(title="Sentinel Deepfake Detection API")

# Initialize Orchestrator (Loads Model)
try:
    orchestrator = Orchestrator()
    logger.info("✅ Core Orchestrator Online")
except Exception as e:
    logger.error(f"❌ Failed to initialize Orchestrator: {e}")
    orchestrator = None # Graceful failure handling

# Environment Setup
# (No external API keys required for local model)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins to prevent CORS/Fetch errors
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve evidence files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.get("/sentinel_client.js")
async def read_js():
    return FileResponse('sentinel_client.js')

@app.get("/health")
def health_check():
    status = "Sentinel System Online" if orchestrator else "Sentinel System Degraded (Model Failed)"
    return {"status": status, "version": "2.0.0"}

@app.post("/scan")
async def scan_video(file: UploadFile = File(...)):
    """
    Endpoint to trigger deepfake scanning. Supports Video and Images.
    """
    if not orchestrator:
        return JSONResponse(status_code=503, content={"error": "Model not initialized."})

    logger.info(f"Received file: {file.filename} ({file.content_type})")
    
    # Save Uploaded File to Temp
    # We use a named temporary file to ensure we have a path to pass to OpenCV
    try:
        suffix = os.path.splitext(file.filename)[1]
        if not suffix:
            # Fallback based on content type
            if "image" in file.content_type:
                suffix = ".jpg"
            else:
                suffix = ".mp4"
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        logger.info(f"File saved to temporary path: {tmp_path}")
        
        # Analyze
        if "image" in file.content_type or suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
             results_obj = orchestrator.process_image(tmp_path)
        else:
             results_obj = orchestrator.process_video(tmp_path)
        
        # Cleanup
        try:
            os.remove(tmp_path)
            logger.info(f"Temporary file removed: {tmp_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temp file: {e}")
            
        return JSONResponse(content=results_obj.dict())

    except Exception as e:
        logger.error(f"Analysis Failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "message": "Analysis failed internally."}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
# Manual Restart Trigger
