from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas import DeepfakeResponse, HealthResponse, Segment

app = FastAPI(title="DeepFake Detector API")

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health", response_model=HealthResponse)
def health_check():
    return {"status": "ok"}

import shutil
import os
from backend.orchestrator import Orchestrator

# Initialize Orchestrator once
orchestrator = Orchestrator()

@app.post("/api/analyze", response_model=DeepfakeResponse)
async def analyze_video(file: UploadFile = File(...)):
    # Save Uploaded File
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Run Analysis
    try:
        result = orchestrator.process_video(temp_path)
        return result
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/analyze/image", response_model=DeepfakeResponse)
async def analyze_image(file: UploadFile = File(...)):
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        result = orchestrator.process_image(temp_path)
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

from fastapi.staticfiles import StaticFiles

# Serve React Frontend (Must be after API routes)
# In Docker, we are at /app, and static files are at /app/backend/static
app.mount("/", StaticFiles(directory="backend/static", html=True), name="static")
