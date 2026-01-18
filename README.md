# Sentinel // Deepfake Detector

**Sentinel** is an advanced AI-powered forensic tool designed to detect deepfake media. It uses a hybrid neural network (**EfficientNet-B4 + LSTM**) to analyze video frames for manipulation artifacts and temporal inconsistencies.

![Sentinel Core](https://img.shields.io/badge/Status-Operational-emerald) ![Docker](https://img.shields.io/badge/Docker-Ready-blue) ![Python](https://img.shields.io/badge/Python-3.9+-yellow)

## Features
- **Hybrid Architecture**: Combines spatial features (EfficientNet) with temporal analysis (LSTM).
- **Frame-by-Frame Forensics**: precision checking of video sequences.
- **Visual Evidence**: Extracts and displays compromised frames with timestamps.
- **Playback Sync**: Click evidence frames to jump to the exact moment in the video.
- **Privacy-First**: All processing happens locally (or in your Docker container). No external APIs required.

---

## 🚀 Quick Start (Docker)
The easiest way to run Sentinel is via Docker. This ensures all dependencies are isolated.

1. **Build the Image**
   ```bash
   docker build -t deepfake-detector .
   ```

2. **Run the Container**
   ```bash
   docker run -p 7860:7860 deepfake-detector
   ```

3. **Access the App**
   Open your browser and navigate to:
   👉 **[http://localhost:7860](http://localhost:7860)**

---

## 🛠️ Local Installation (Python)

If you prefer to run it directly on your machine (Mac/Linux/Windows), follow these steps.

### Prerequisites
- Python 3.9 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/suryawanshiheramb11/auraverse2.git
cd auraverse2
```

### 2. Setup Virtual Environment (Recommended)
```bash
# MacOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*Note: This will install PyTorch, TorchVision, FastAPI, Uvicorn, and other core libraries.*

### 4. Verify Model
Ensure `sentinel_model.pth` is in the root directory. This file contains the trained weights for the SentinelHybrid model.

### 5. Run the Server
```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```
*Note: In local mode, the app usually runs on port **8000**.*

### 6. Access the App
Open: **[http://localhost:8000](http://localhost:8000)**

---

## 📂 Project Structure

```
├── backend/
│   ├── orchestrator.py    # Core analysis pipeline (Model Loading, Inference)
│   ├── schemas.py         # Data models (API Responses)
│   └── processing/        # Frame extraction utilities
├── static/
│   └── evidence/          # Extracted fake frames are saved here
├── Dockerfile             # Container configuration
├── server.py              # FastAPI server entry point
├── index.html             # Main UI (Sentinel V2.5 Blue Theme)
├── sentinel_client.js     # Frontend Logic & API Client
├── model_core.py          # PyTorch Model Architecture (EfficientNet+LSTM)
└── requirements.txt       # Python dependencies
```

## 🐛 Troubleshooting

**"Failed to Fetch" Error**
- Ensure the server is running.
- In Docker, make sure you mapped the ports correctly (`-p 7860:7860`).
- Check the terminal logs for backend errors.

**Model Not Found**
- The system looks for `sentinel_model.pth` in the root. Verify it exists.

**Slow Performance / Long Videos**
- Deepfake detection is compute-intensive. Running on CPU (especially in Docker) will be slower than native execution.
- **Smart Sampling**: For videos longer than ~50 seconds, Sentinel automatically uses "Block Sampling" to analyze distributed segments across the video duration, ensuring fast results without timeouts.vely on a GPU-enabled machine (e.g., Mac M-series with MPS).

---

> **Sentinel Initiative** // Protecting Digital Integrity
