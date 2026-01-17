# Technical Retrospective & Strategic Roadmap (Deepfight V1 -> V2)

**Document Version**: 2.0
**Date**: January 16, 2026
**Scope**: Technical Architecture, Failure Analysis, Benchmarking, and Future Engineering.

---

## 1. Architectural Evolution & Failure Analysis

Our journey from a client-side toy to a hybrid AI system was defined by three distinct architectural failures. Each failure provided the constraints for the next iteration.

### Phase I: The Biological Fallacy (Client-Side Heuristics)
*   **Hypothesis**: Deepfakes lack physiological micro-signals (blinking, pulse).
*   **Implementation**: `face-api.js` (TensorFlow.js) running in the browser.
*   **Technical Failure**:
    *   **Adversarial Robustness**: Zero. The "Blink" heuristic is trivial to bypass. Generative Adversarial Networks (GANs) like **StyleGAN2** and **Wav2Lip** optimize specifically to minimize this loss function.
    *   **Latency**: Client-side inference on high-resolution video caused Main Thread blocking (UI freeze), degrading UX.

### Phase II: The Mathematical Fallacy (Frequency Analysis)
*   **Hypothesis**: GAN upsampling operations (Transposed Convolutions) leave "checkerboard artifacts" in the frequency domain.
*   **Implementation**: Python/OpenCV performing Fast Fourier Transform (FFT) and Laplacian Variance analysis.
*   **Technical Failure**:
    *   **Signal-to-Noise Ratio (SNR)**: In uncompressed video (BMP/PNG), this works. In H.264/MP4 video, the quantization step of the CODEC acts as a low-pass filter, obliterating the high-frequency "checkerboard" artifacts.
    *   **Result**: 80%+ False Positive Rate on social media video (WhatsApp/Instagram).

### Phase III: The Black Box Integration (Current State)
*   **Hypothesis**: A pre-trained Vision Transformer (ViT) can detect semantic inconsistencies better than humans.
*   **Implementation**: `DeepFake-Detector-v2.onnx` (FP16/Quantized).
*   **Critical Engineering Errors**:
    1.  **Color Space Alignment**:
        *   *Error*: Fed standard `RGB` (0-255) to a model trained on `BGR` (Caffe-style).
        *   *Consequence*: Channel indices were swapped. Red lips were interpreted as Blue. Skin tone distributions were inverted. The model drifted into undefined latent space behavior.
    2.  **Logits Interpretation (Polarity)**:
        *   *Error*: Assumed output vector `[x0, x1]` mapped to `[Real, Fake]`.
        *   *Reality*: Model specific mapping was `[Fake, Real]`.
        *   *Consequence*: Inverted accuracy (Accuracy < 50% implies systematic inversion).
    3.  **Graph Optimization Conflicts**:
        *   *Error*: `SimplifiedLayerNormFusion` in ONNX Runtime conflicted with the specific opset version of the exported model.
        *   *Resolution*: Forced `ORT_DISABLE_ALL` to bypass fusion kernels.

---

## 2. Technical Benchmarks & Metrics (The Standard)

To claim "Robustness", the system must be evaluated against standard datasets (FaceForensics++, Celeb-DF). We currently lack a rigorous testing pipeline.

### Required Metrics for V2
A "working" detector is not measured by "It detected Tom Cruise". It is measured by:

1.  **AUC (Area Under the Receiver Operating Characteristic Curve)**
    *   *Target*: **> 0.95**.
    *   *Definition*: The ability to distinguish Fake from Real across *all* threshold settings. A classifier with AUC 0.5 is random guessing.

2.  **EER (Equal Error Rate)**
    *   *Target*: **< 5%**.
    *   *Definition*: The threshold where `False Acceptance Rate (FAR)` == `False Rejection Rate (FRR)`. Lower is better. This is the "Security" metric.

3.  **LogLoss (Cross-Entropy Loss)**
    *   *Target*: **< 0.2**.
    *   *Definition*: Measures confidence. A model that says "51% Fake" for a Fake is worse than one that says "99% Fake".

4.  **Robustness to Perturbation**
    *   **Gaussian Blur**: Sigma 1.0 - 5.0.
    *   **JPG Compression**: Quality 90 down to 50.
    *   **Gaussian Noise**: Sigma 1.0 - 10.0.
    *   *Test*: Does accuracy drop by >10% under these conditions? If yes, the model is brittle.

---

## 3. The "Defense-in-Depth" Specification (V2 Architecture)

Version 1 is a "Cascading Gate". Version 2 must be an **Ensemble**.

### Component A: The Spatial Expert (EfficientNet-B7)
*   **Architecture**: EfficientNet (Compound Scaling).
*   **Training Strategy**:
    *   **Dataset**: FaceForensics++ (Raw + C23 + C40 compression levels).
    *   **Augmentation**: CutOut, RandomErasing, and JpegCompression (crucial for WhatsApp robustness).
*   **Role**: Detects pixel-level artifacts in single frames.

### Component B: The Temporal Expert (LSTM / 3D-CNN)
*   **Architecture**: ResNet + LSTM or I3D (Inflated 3D ConvNet).
*   **Input**: Sequence of 30 frames (1 second).
*   **Role**: Detects **Temporal Flicker**. GANs generate frames independently, often causing lighting/texture to "flicker" at 30Hz. Single-frame models (like our current V1) are blind to this.

### Component C: The Audio-Visual Expert (SyncNet)
*   **Architecture**: Dual-stream network (Audio Stream + Video Consumer).
*   **Logic**: Measures the distance between the *phoneme* (sound) and the *viseme* (mouth shape).
*   **Role**: Detects **Lip-Sync Errors**. Wav2Lip models are good, but often have a 50-100ms drift. Humans perceive this subconsciously; detection models can quantify it.

### Component D: The Fusion Layer (Calibrated Head)
*   **Logic**: Instead of a simple weighted average (`0.6*AI + 0.4*Forensics`), train a **Logistic Regression** or **XGBoost** model to combine the scores of Experts A, B, and C.
*   **Why**: The "Spatial Expert" might be wrong on dark videos, but the "Audio expert" is confident. A learned fusion layer knows which expert to trust in which scenario.

---

## 4. Immediate Action Items (The Path Forward)

1.  **Data Acquisition**: Download **FaceForensics++** (50GB). This is non-negotiable for serious work.
2.  **Training Pipeline**: Setup a PyTorch training loop with `Albumentations` for heavy compression augmentation.
3.  **Evaluation Harness**: Write a script that runs the model against a held-out test set and calculates **EER** and **AUC** automatically. No more "eyeballing" terminal logs.
