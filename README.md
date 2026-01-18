<div align="center">
  <img src="image_38933a.png" alt="Sentinel Spatial Intelligence Engine" width="100%" />

  <h1>🛡️ SENTINEL</h1>
  <h3>Spatial Intelligence Engine // Deepfake Forensics</h3>

  <p>
    <a href="https://fuseless-wynell-unspasmodical.ngrok-free.dev/">
      <img src="https://img.shields.io/badge/System-Operational-emerald?style=for-the-badge&logo=statuspage" alt="System Operational" />
    </a>
    <a href="https://github.com/suryawanshiheramb11/auraverse2/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge&logo=open-source-initiative" alt="License MIT" />
    </a>
    <a href="https://python.org">
      <img src="https://img.shields.io/badge/Python-3.9+-yellow?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.9+" />
    </a>
    <a href="https://www.docker.com/">
      <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker Ready" />
    </a>
  </p>

  <p>
    <a href="https://fuseless-wynell-unspasmodical.ngrok-free.dev/">
      <img src="https://img.shields.io/badge/🚀_Launch_Live_Demo-000000?style=for-the-badge&logo=vercel&logoColor=white" alt="Launch Live Demo" />
    </a>
  </p>
</div>

---

## 📖 Overview

**Sentinel** is an advanced AI-powered forensic tool designed to safeguard digital integrity. Utilizing a hybrid neural network architecture (**EfficientNet-B4 + LSTM**), Sentinel analyzes video media frame-by-frame to detect manipulation artifacts and temporal inconsistencies typical of deepfakes.

> *"In an era of synthetic media, truth requires a Sentinel."*

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **🧠 Hybrid Intelligence** | Combines **Spatial analysis** (EfficientNet) with **Temporal analysis** (LSTM) for high-accuracy detection. |
| **🔍 Micro-Forensics** | Performs precision checking on video sequences to detect pixel-level anomalies. |
| **⚡ Real-Time Evidence** | Extracts compromised frames with exact timestamps and confidence scores. |
| **📼 Smart Playback** | Click on any evidence frame to instantly jump to that exact moment in the video player. |
| **🔒 Privacy First** | All processing happens locally or within your Docker container. No external APIs required. |

---

## 🛠️ Tech Stack

* **Core:** ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
* **Backend:** ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) ![Uvicorn](https://img.shields.io/badge/Uvicorn-499848?style=flat-square&logo=gunicorn&logoColor=white)
* **Frontend:** ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)
* **Deployment:** ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)

---

## 🚀 Quick Start (Docker)

The fastest way to deploy Sentinel is via Docker. This ensures total isolation and consistent dependencies.

1.  **Build the Image**
    ```bash
    docker build -t deepfake-detector .
    ```

2.  **Run the Container**
    ```bash
    docker run -p 7860:7860 deepfake-detector
    ```

3.  **Access the System**
    Open your browser and navigate to:
    👉 **[http://localhost:7860](http://localhost:7860)**

---

## ⚙️ Local Installation (Python)

If you prefer native execution (Mac/Linux/Windows), follow these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/suryawanshiheramb11/auraverse2.git](https://github.com/suryawanshiheramb11/auraverse2.git)
cd auraverse2

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
