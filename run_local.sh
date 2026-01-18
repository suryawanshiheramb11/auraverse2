#!/bin/bash
echo "Stopping any existing processes..."
pkill -f uvicorn
pkill -f "python server.py"

echo "Installing Dependencies..."
pip3 install -r requirements.txt

echo "Starting Sentinel Server (Local Mode)..."
python3 -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
