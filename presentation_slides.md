---
marp: true
theme: default
html: true
style: |
  section {
    /* "Animal" nature background: Warm orange/sand to forest/sage green */
    background: linear-gradient(135deg, #F4A261 0%, #E9C46A 50%, #2A9D8F 100%);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    padding: 60px 80px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    color: #333333;
  }
  
  /* Glassmorphism Container */
  .glass-card {
    background: rgba(255, 255, 255, 0.25);
    backdrop-filter: blur(25px);
    -webkit-backdrop-filter: blur(25px);
    border: 1px solid rgba(255, 255, 255, 0.5);
    border-radius: 24px;
    padding: 35px 50px;
    width: 92%;
    /* Fixed shadow: much softer, clean drop shadow without the dirty blur */
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
    text-align: left;
  }

  .glass-card-center {
    text-align: center;
  }

  h1 {
    font-size: 2.3em;
    font-weight: 700;
    margin: 0 0 0.2em 0;
    color: #111111;
    text-shadow: none; /* Fixed dirty text shadow */
  }

  h2 {
    font-size: 1.2em;
    font-weight: 400;
    color: #444444;
    margin: 0 0 0.8em 0;
    border: none;
  }

  p, li {
    font-size: 1.05em;
    font-weight: 400;
    line-height: 1.5;
    color: #222222;
    margin-bottom: 8px;
  }

  strong {
    font-weight: 700;
    color: #D35400; /* Deep terracotta/animal orange accent */
  }
---

<div class="glass-card glass-card-center">

# Pet Translator
## Bridging the Gap Between Pets and Humans

<br><br>

<p><strong>Team:</strong> Amine KHALIL • Mohamed MELLOUK • Anas ISARTI • Abir ISLAM</p>
<p>Machine Learning (ML01)</p>

</div>

---

<div class="glass-card">

# Architecture
## An End-to-End Hybrid Pipeline

- **Frontend:** Real-time audio capture via React.
- **Edge Inference:** Local, ultra-fast classification on the smartphone via **TensorFlow.js**.
- **Cloud Backend:** FastAPI server connecting to a semantic database.
- **Generative AI:** **Llama 3.2** generating contextual human sentences.

</div>

---

<div class="glass-card">

# Audio Signal Processing
## Mohamed MELLOUK

- **Datasets:** ESC-50, AudioSet, ShivaRao (Dogs), CatMeows (Cats).
- **The Challenge:** Dog sounds are *acoustic types* (bark). Cat sounds are *situational* (food).
- **Noise Reduction:** Applying `noisereduce` to clean real-world ambient sounds.
- **Isolation:** Using Voice Activity Detection (WebRTCVAD) to isolate vocalization.

</div>

---

<div class="glass-card">

# Feature Extraction
## Visualizing Sound

- Audio waves are converted into visual representations (**Mel-spectrograms**).
- **Standardization:** Fixed durations of 4 seconds for dogs and 2 seconds for cats.
- **Perception:** Logarithmic scaling applied on Mel frequencies to mimic biological hearing.
- **Result:** 96x96 feature matrices, formatted for Deep Learning.

</div>

---

<div class="glass-card">

# Classification Models
## Anas ISARTI

- **Task:** 2 classifiers, 1 par animal. Dog: bark/growl/grunt. Cat: brushing/food/isolation.
- **Challenge:** Tiny datasets (113 dog, 440 cat clips) → need a model that generalizes + an evaluation I can trust.
- **Approach:** Frozen **MobileNetV2** (feature extractor) + small trained head. Avoids overfitting vs training from scratch.
- **Evaluation:** Group-aware cross-validation (split by individual cat) → no data leakage.
- **Deployment:** Full model exported to **TensorFlow.js**, output verified identical to Python (diff = 0).

</div>

---

<div class="glass-card">

# The Data Ceiling
## Analyzing Limits

- **Dog model:** Solid — CV macro-F1 ~0.82–0.85.
- **Cat model:** Good on isolation, weak on food — CV F1 ~0.30–0.37.
- **Experimentation:** 5 independent levers tested (head tuning, augmentation, classifier comparison, AST backbone, focal loss + SMOTE) — all plateaued.
- **Conclusion:** Bottleneck is the dataset size (food: only 92 clips), not model capacity.

</div>

---

<div class="glass-card">

# Linguistic Brain
## Abir ISLAM

- **Objective:** Give a real voice and distinct personality to the pet's raw intent.
- **Dataset:** Created a synthetic dataset mapping `[Intent + Context + Personality]` to `[Sentence]`.
- **Fine-Tuning:** Trained **Llama 3.2** using LoRA on Google Colab GPUs.
- **Deployment:** Converted weights to GGUF format for lightning-fast CPU inference.

</div>

---

<div class="glass-card">

# Grounding the AI
## The RAG System

- **The Problem:** LLMs can hallucinate biologically inaccurate translations.
- **The Solution:** Retrieval-Augmented Generation (RAG).
- **Implementation:**
  - Built a veterinary behavioral knowledge base.
  - Embedded using `sentence-transformers` and stored in **ChromaDB**.
  - The LLM retrieves real scientific context before generating the translation.

</div>

---

<div class="glass-card">

# Frontend Integration
## Amine KHALIL

- **User Interface:** A modern, responsive React application with dark/light theme, three-column layout, and chat-bubble translations.
- **Pet Selector:** Allows selecting the animal (cat/dog) - personality is assigned server-side per species (haughty_cat / excited_dog) and not exposed in the UI.
- **Real-Time Feedback:** TFJS in-browser classification computes class probabilities (bark, growl, grunt / brushing, food, isolation) and displays them alongside the LLM-translated output.

</div>

---

<div class="glass-card">

# Edge Deployment & UX
## Amine KHALIL

- **TF.js Deployment:** Converted Keras model to GraphModel format; fixed 4 preprocessing bugs (mel scale, filter norm, power vs magnitude, input range) to match Python exactly.
- **VAD:** Energy-based silence detection (RMS < 0.015) prevents classification on silent audio, returning "no sound detected".
- **Model Loading:** Explicit `tf.setBackend('webgl').catch(() => 'cpu')` required before model load to prevent silent fails on macOS.
- **UX Research:** 3-second minimum loading delay for perception; probability bars; mock LLM fallback ensures demo works offline.

</div>

---

<div class="glass-card glass-card-center">

# Demonstration

<br><br><br>

*(Live Video Demonstration)*

</div>

---

<div class="glass-card">

# Conclusion
## 

- **End-to-End Pipeline:** Successfully integrated Edge AI (TFJS) for low-latency classification with Cloud AI (Llama 3.2 + RAG) for linguistic grounding.
- **Scientific Rigor:** Demonstrated the critical impact of dataset size ("Data Ceiling") through exhaustive evaluation and cross-validation.
- **Robust Architecture:** Solved real-world constraints like ambient noise (VAD) and LLM hallucinations (RAG).
- **Final Result:** A fully functional WebApp simulating a mobile application, bridging audio signal processing, Edge classification, and NLP.

<br>
<p><strong>Thank you.</strong></p>

</div>
