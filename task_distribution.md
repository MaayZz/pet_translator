# Task Distribution (Team of 4 Students)

This document details the specific responsibilities for each team member regarding the **Pet Translation Device** project. The core LLM chosen for natural language generation is **Llama 3.2**.

## Student 1: Mohamed MELLOUK - Audio Signal Processing and VAD
**Role:** Responsible for the capture, cleaning, and preparation of raw audio data.
*   **Tasks:**
    *   **Data Collection:** Set up the audio acquisition pipeline (e.g., using existing pet datasets like ESC-50 or real recordings).
    *   **Noise Reduction:** Develop and test ambient noise reduction algorithms (denoising) to filter out background noise (wind, traffic, etc.).
    *   **VAD Implementation:** Implement Voice Activity Detection (VAD) to precisely isolate the start and end of the animal's vocalizations.
    *   **Pipeline Optimization:** Ensure the audio preprocessing pipeline is efficient and introduces minimal latency.
*   **Estimated Workload:** 25%

## Student 2: Anas ISARTI - Classification Models & Interpretation
**Role:** Responsible for the analytical "brain" that translates raw audio into an emotional category or intent.
*   **Tasks:**
    *   **Model Selection:** Select, train, or fine-tune an audio classification model (e.g., Audio Spectrogram Transformer, CNNs on Mel-spectrograms).
    *   **Intent Mapping:** Define a clear ontology of intent categories (e.g., Hunger, Pain, Play, Threat, Greeting).
    *   **Evaluation & Metrics:** Rigorously evaluate model performance using Precision, Recall, and F1-score. Handle false positives to avoid confusing the user.
    *   **Integration Preparation:** Export the trained model in a format suitable for Edge AI deployment (collaboration with Amine).
*   **Estimated Workload:** 25%

## Student 3: Abir ISLAM - LLM Interaction & Personality Generation
**Role:** Responsible for the human-like "translation" using **Llama 3.2**.
*   **Tasks:**
    *   **LLM Integration:** Integrate **Llama 3.2** (running locally via tools like Ollama/llama.cpp, or via API) to convert the classified intent into natural, contextual sentences.
    *   **Prompt Engineering:** Design system prompts to inject a specific persona into the translation (e.g., a haughty cat, a goofy dog, or an overly dramatic pet).
    *   **Context Management:** Generate and maintain chat history to give the illusion of an ongoing, continuous conversation rather than isolated messages.
    *   **Trade-off Analysis:** Analyze the latency and resource consumption of Llama 3.2, comparing local execution vs. remote API inference, to provide the "deeper insights" requested by the professor.
*   **Estimated Workload:** 25%

## Student 4: Amine KHALIL - Edge AI Deployment & Mobile Application
**Role:** Responsible for final system integration, performance optimization, and user experience.
*   **Tasks:**
    *   **Model Optimization:** Optimize the audio preprocessing and classification models to run on Edge devices (e.g., quantization to INT8, conversion to TFLite or CoreML).
    *   **UI/UX Design:** Create a simple user interface (mobile application or interactive mock-up) simulating the smartphone screen, presenting the "chat" with the pet.
    *   **System Integration:** Manage the asynchronous communication between the audio stream, the classification model, and the Llama 3.2 backend.
    *   **Latency Benchmarking:** Measure and document the end-to-end latency from a bark/meow to the text appearing on the screen.
*   **Estimated Workload:** 25%
