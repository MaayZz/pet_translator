from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from llm.generator import PetTranslatorLLM, INTENT_DESCRIPTIONS

app = FastAPI(title="Pet Translator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

history_store = []

DEFAULT_PERSONALITY = {"cat": "haughty_cat", "dog": "excited_dog"}
llm = PetTranslatorLLM()

class TranslateRequest(BaseModel):
    animal: str = "cat"
    label: str = "uncertain"
    confidence: float = 0.0
    probabilities: dict = {}
    history: list = []

class TranslateResponse(BaseModel):
    text: str
    emotion: str
    confidence: float
    timestamp: str

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    personality = DEFAULT_PERSONALITY.get(req.animal, "haughty_cat")
    intent_desc = INTENT_DESCRIPTIONS.get(req.animal, {}).get(req.label, req.label)
    text = llm.translate(
        intent_category=intent_desc,
        intent_label=req.label,
        personality=personality,
        confidence=req.confidence,
        probabilities=req.probabilities
    )
    now = datetime.now().isoformat()
    entry = {
        "text": text,
        "emotion": req.label,
        "confidence": req.confidence,
        "animal": req.animal,
        "personality": personality,
        "timestamp": now,
    }
    history_store.append(entry)
    return TranslateResponse(
        text=text,
        emotion=req.label,
        confidence=req.confidence,
        timestamp=now,
    )

@app.get("/history")
def get_history():
    return history_store[-50:]
