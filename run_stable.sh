#!/bin/bash
echo "Stopping any existing processes..."
pkill -f uvicorn
pkill -f "python server.py"

echo "Starting Sentinel Server (Stable Mode)..."
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
