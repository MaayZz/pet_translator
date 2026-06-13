# Task Distribution (Team of 4 Students)

This document details the specific responsibilities for each team member regarding the **Pet Translation Device** project.

## Student 1: Audio Signal Processing and VAD (Voice Activity Detection)
**Role:** Responsible for the capture and preparation of raw audio data.
*   **Tasks:**
    *   Set up the audio acquisition pipeline (existing pet datasets or real recordings).
    *   Develop ambient noise reduction (denoising).
    *   Implement a VAD to isolate exactly when the animal emits a sound.
*   **Estimated Workload:** 25%

## Student 2: Classification Models & Interpretation (Core ML)
**Role:** Responsible for the analytical "brain" that gives meaning to the sound.
*   **Tasks:**
    *   Select and train or fine-tune an audio classification model.
    *   Create intent categories (e.g., Hunger, Pain, Play, Threat).
    *   Evaluate model performance (Precision, Recall) and handle false positives.
*   **Estimated Workload:** 25%

## Student 3: LLM Interaction & Personality Generation
**Role:** Responsible for the human "translation".
*   **Tasks:**
    *   Integrate an LLM (e.g., via API or local model) to convert the classification category into a natural and amusing sentence.
    *   Prompt engineering to give a specific personality to the pet (e.g., haughty cat, overly excited dog).
    *   Generate chat history to give the illusion of a real conversation.
*   **Estimated Workload:** 25%

## Student 4: Edge AI Deployment & Mobile Application
**Role:** Responsible for final integration and user experience.
*   **Tasks:**
    *   Optimize models to run locally as much as possible (quantization, conversion to mobile formats).
    *   Create a simple user interface (Application or interactive mock-up) simulating the smartphone screen (the "chat" with the pet).
    *   Manage communication between the audio stream, the local model, and the LLM call.
*   **Estimated Workload:** 25%
