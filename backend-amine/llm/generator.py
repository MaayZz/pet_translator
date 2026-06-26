import os
import random
from .prompt import get_system_prompt, build_user_prompt

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "unsloth.Q4_K_M.gguf")

INTENT_DESCRIPTIONS = {
    "dog": {
        "bark": "the dog is barking loudly — urgent, alerting, or excited",
        "growl": "the dog is growling — warning, discomfort, or threat",
        "grunt": "the dog is grunting — mild annoyance or contentment",
        "uncertain": "the pet made an unclear sound — best guess with caution",
    },
    "cat": {
        "brushing": "the cat is enjoying being brushed — pleasure, trust",
        "food": "the cat is meowing for food — hunger, impatience",
        "isolation": "the cat is crying from being alone — distress, loneliness",
        "uncertain": "the pet made an unclear sound — best guess with caution",
    },
}

MOCK_RESPONSES = {
    "dog": {
        "bark": {
            "excited_dog": "WOOF WOOF! Something's happening! Let's go!",
            "shy_dog": "Um, there's something... I think we should check?",
        },
        "growl": {
            "excited_dog": "Grrr... stay back! I don't like this!",
            "shy_dog": "Please... stay away from me...",
        },
        "grunt": {
            "excited_dog": "Hmph. Fine. Whatever. Are we playing or not?",
            "shy_dog": "I'm okay. Just resting.",
        },
        "uncertain": {
            "excited_dog": "I'm trying to tell you something important!",
            "shy_dog": "I... I have something to say... maybe...",
        },
    },
    "cat": {
        "brushing": {
            "haughty_cat": "Mmm. You may continue. I shall allow it.",
            "grumpy_cat": "Fine. But don't think this means I like you.",
        },
        "food": {
            "haughty_cat": "My bowl is empty. This is an outrage.",
            "grumpy_cat": "Food. Now. Don't make me repeat myself.",
        },
        "isolation": {
            "haughty_cat": "You left me alone? How dare you.",
            "grumpy_cat": "Where is everyone? This is unacceptable.",
        },
        "uncertain": {
            "haughty_cat": "I expect you to understand what I want.",
            "grumpy_cat": "Figure it out. I'm not explaining myself.",
        },
    },
}

from .rag_retriever import RAGRetriever

class PetTranslatorLLM:
    def __init__(self):
        self.llm = None
        self.rag = RAGRetriever()
        self._load_model()

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            print(f"Warning: Model not found at {MODEL_PATH}. Using mock responses.")
            return
        try:
            from llama_cpp import Llama
            print("Loading GGUF Llama model...")
            self.llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=2048,
                n_gpu_layers=-1,
            )
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}. Using mock responses.")

    def translate(self, intent_category, personality="excited_dog", env_context=None, confidence=None, probabilities=None, intent_label=None):
        if self.llm is not None:
            try:
                system_prompt = get_system_prompt(personality)
                rag_context = self.rag.retrieve_context(f"What does {intent_category} mean?")
                user_prompt = build_user_prompt(intent_category, rag_context, env_context, confidence, probabilities)
                response = self.llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=60,
                    temperature=0.7,
                )
                return response["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"LLM inference failed: {e}")

        return self._mock_response(intent_label or intent_category, personality)

    def _mock_response(self, intent_category, personality):
        pet_key = "dog" if "dog" in personality else "cat"
        pet_mock = MOCK_RESPONSES.get(pet_key, {})
        intent_mock = pet_mock.get(intent_category, {})
        if intent_mock:
            return intent_mock.get(personality, list(intent_mock.values())[0])
        for v in pet_mock.values():
            if personality in v:
                return v[personality]
        return "Woof."
