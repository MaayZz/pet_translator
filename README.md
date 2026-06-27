# Pet Translator 🐾

## Description
This project is an end-to-end "Pet Translation Device" concept developed for the Machine Learning (ML01) course. It captures pet audio (cats/dogs), analyzes the emotional intent directly in the browser via Edge AI, and uses a Large Language Model (LLM) grounded with RAG to translate it into a natural language sentence.

## Team
- **Amine KHALIL** - Frontend Web, UI/UX & Full Integration
- **Mohamed MELLOUK** - Audio Signal Processing & Pipeline
- **Anas ISARTI** - Classification Models (MobileNetV2 to TFJS) & Scientific Evaluation
- **Abir ISLAM** - LLM Fine-Tuning, RAG & Inference (Llama 3.2 + LoRA)

## Documentation
- `docs/abir_research_report.md` & `docs/mohamed_research_report.md` : Research justifications and methodology.
- `docs/task_distribution.md` : Detailed workload and tasks per member.
- `presentation_slides.pdf` : Final presentation slides.

## Setup & Execution

### 1. Frontend (React + Vite + TF.js)
The frontend captures audio, handles Voice Activity Detection (VAD), and runs the Edge AI classification model.

```bash
cd frontend-amine
npm install
npm run dev
```
*Note: Make sure to allow microphone permissions in your browser.*

### 2. Backend (FastAPI + LLM + RAG)
The backend acts as the linguistic brain, translating the predicted intent into a human sentence.

```bash
cd backend-amine
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn main:app --reload
```
*Note: The heavy LLM weights (.gguf) are excluded from this repository due to GitHub size limits, but are required to run the local backend.*

## Architecture Highlights
- **Edge Inference:** TensorFlow.js running WebGL directly in the browser for ultra-low latency.
- **VAD (Voice Activity Detection):** Energy-based silence detection (RMS < 0.015) preventing classification on silent audio.
- **LLM Engine:** Llama 3.2 fine-tuned with LoRA, served locally via `llama.cpp` and FastApi.
- **RAG System:** ChromaDB embedding veterinary knowledge to scientifically ground the generated translations.
