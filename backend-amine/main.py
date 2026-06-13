from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from llm.generator import generate_response
from llm.prompt import build_prompt

app = FastAPI(title="Pet Translator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

history_store = []

DEFAULT_PERSONALITY = {"cat": "cat_snobbish", "dog": "dog_excited"}

class TranslateRequest(BaseModel):
    category: str
    confidence: float
    pet_type: str = "cat"
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
    personality = DEFAULT_PERSONALITY.get(req.pet_type, "cat_snobbish")
    prompt = build_prompt(
        category=req.category,
        confidence=req.confidence,
        pet_type=req.pet_type,
        personality=personality,
        history=req.history,
    )
    text = generate_response(prompt)
    now = datetime.now().isoformat()
    entry = {
        "text": text,
        "emotion": req.category,
        "confidence": req.confidence,
        "pet_type": req.pet_type,
        "personality": personality,
        "timestamp": now,
    }
    history_store.append(entry)
    return TranslateResponse(
        text=text,
        emotion=req.category,
        confidence=req.confidence,
        timestamp=now,
    )

@app.get("/history")
def get_history():
    return history_store[-50:]
