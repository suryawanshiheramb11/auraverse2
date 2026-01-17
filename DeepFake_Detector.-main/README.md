---
title: DeepFake Detector
emoji: 🕵️
colorFrom: gray
colorTo: red
sdk: docker
pinned: false
---
# DeepForged Forensics: Temporal DeepFake Detector

A professional-grade forensic tool designed to detect manipulation in digital media. This system not only identifies *if* a video is fake but specifically identifies *where* (temporally) the manipulation occurs.

## 🚀 Live Demo
**[Launch App on Hugging Face Spaces](https://huggingface.co/spaces/sid2713/DeepFake_Detector)**

## ⚡ Key Features
*   **Multi-Modal Analysis**: Supports both **Video** (MP4, AVI, MOV) and **Image** (JPG, PNG) inputs.
*   **Temporal Localization**: Instead of a single "Fake" label, we provide a **timeline** showing exactly which seconds of a video are manipulated.
*   **Forensic Dashboard**:
    *   Interactive Video Player with "Red Zone" navigation.
    *   Confidence scoring per segment.
    *   **JSON Export** for forensic reporting.
*   **Privacy First**: No data is permanently stored. Files are processed in memory and wiped effectively after analysis.

## 🛠️ Architecture
The system is built on a decoupled, scalable architecture:
*   **Frontend**: React 19 + Vite + TailwindCSS (Single Page Application).
*   **Backend**: FastAPI (Python) with asyncio for high-concurrency processing.
*   **Engine**: OpenCV for frame extraction + PyTorch (Ready) for inference.
*   **Deployment**: Dockerized multi-stage build (Node.js -> Python) on Hugging Face Spaces.

## ⚠️ Current Status: Prototype Mode
**Note:** This deployment is currently running in **Logic Verification Mode**.
*   The system uses a **Mock Expert** engine to demonstrate the pipeline, UI, and reporting capabilities.
*   It generates *simulated* detection scores to fully validate the frontend interaction and backend orchestration.
*   **Real Model Integration**: The architecture is "Plug-and-Play" ready for the final trained model weights.

## 🏃‍♂️ How to Run Locally
1.  **Clone the repo**:
    ```bash
    git clone https://github.com/YourUsername/DeepFakeDetector.git
    cd DeepFakeDetector
    ```
2.  **Run with Docker (Recommended)**:
    ```bash
    docker build -t deepfake-detector .
    docker run -p 7860:7860 deepfake-detector
    ```
3.  **Access**: Open `http://localhost:7860`.

---
*Developed for the DeepFake Detection Architecture Challenge.*
