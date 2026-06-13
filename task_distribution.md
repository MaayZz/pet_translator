# Task Distribution (Team of 4 Students)

This document details the specific responsibilities for each team member regarding the **Pet Translation Device** project, precisely aligned with the technical stack outlined in `agents_ia.md`.

## Student 1: Mohamed MELLOUK - Audio Signal Processing and Pipeline
**Role:** Responsible for the capture, cleaning, and extraction of raw audio features.
*   **Target Directories:** `backend/audio/`, `training/classification/data/`
*   **Technologies:** Python, `librosa`, `noisereduce`, `webrtcvad`
*   **Tasks:**
    *   **Data Collection:** Gather and prepare audio datasets (ESC-50, subset of AudioSet, and custom recordings).
    *   **Preprocessing Pipeline:** Implement ambient noise reduction (`noisereduce`) and Voice Activity Detection (`webrtcvad`).
    *   **Feature Extraction:** Convert raw audio waves into Mel-spectrograms and MFCCs using `librosa` to feed the classification model.

## Student 2: Anas ISARTI - Classification Models (MobileNetV2 to TFJS)
**Role:** Responsible for the analytical "brain" that translates raw audio features into an emotional category directly in the browser.
*   **Target Directories:** `training/classification/`, `frontend/public/model/`, `frontend/src/lib/`
*   **Technologies:** TensorFlow/Keras, TensorFlow.js, MobileNetV2
*   **Tasks:**
    *   **Model Fine-Tuning:** Fine-tune a MobileNetV2 architecture on the spectrograms to classify intentions (Hunger, Pain, Play, Attention, Fear, Content).
    *   **Edge Optimization:** Convert the trained model (`.h5`) to `.tflite` and then to TensorFlow.js format with INT8 quantization.
    *   **Browser Integration:** Deploy the TFJS model to the frontend (`modelLoader.js`) to ensure low latency (~20-50ms) local inference.

## Student 3: Abir ISLAM - LLM Fine-Tuning & Inference (Llama 3.2 + LoRA)
**Role:** Responsible for generating human-like "translations" with customized pet personalities.
*   **Target Directories:** `training/llm/`, `backend/llm/`
*   **Technologies:** Llama 3.2 (1B/3B), `unsloth`, `llama.cpp` (Python bindings)
*   **Tasks:**
    *   **Dataset Generation:** Create a synthetic dataset (`synthetic_dataset.json`) of 5000+ pairs mapping {intent + context + personality} to natural language sentences.
    *   **Model Fine-Tuning:** Execute LoRA fine-tuning on Llama 3.2 using `unsloth` and `peft`.
    *   **Deployment & Prompting:** Convert the fine-tuned model to GGUF (Q4_K_M) format for efficient serving via `llama.cpp`. Handle prompt engineering to establish personalities (e.g., haughty cat, excited dog).

## Student 4: Amine KHALIL - Frontend Web, Backend API & Full Integration
**Role:** Responsible for user experience, API routing, and the final end-to-end communication of the pipeline.
*   **Target Directories:** `frontend/`, `backend/main.py`
*   **Technologies:** React + Vite, FastAPI, MediaRecorder API
*   **Tasks:**
    *   **Frontend UI:** Build the React web interface (Chat UI, Audio Recorder, Pet Selector) using MediaRecorder API to capture browser audio.
    *   **Backend Server:** Develop the FastAPI server (`main.py`) to expose the `/translate` endpoint handling requests from the frontend to the Llama inference engine.
    *   **Integration & Deliverables:** Ensure smooth communication from audio capture → TFJS classification → LLM generation → Chat UI. Coordinate the creation of the 10-minute demo video and technical report.
