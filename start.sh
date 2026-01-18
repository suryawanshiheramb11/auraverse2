#!/bin/bash

echo "=========================================="
echo "   SENTINEL // DEEPFAKE FORENSICS SYSTEM  "
echo "=========================================="
echo "Initialize Cyberpunk Forensics on Port 8000"

# Check if requirements are satisfied (Optional hint)
# pip install -r requirements.txt

echo "[SYSTEM] Launching Uvicorn Server..."
export OMP_NUM_THREADS=1
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
