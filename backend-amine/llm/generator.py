import os
import random
from .prompt import get_system_prompt, build_user_prompt

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "unsloth.Q4_K_M.gguf")

MOCK_RESPONSES = {
    "cat": {
        "haughty_cat": {
            "food": "Finally, you decide to feed your king.",
            "brushing": "You may pet me, I grant you this favor.",
            "isolation": "There is an intruder in my kingdom.",
        },
        "grumpy_cat": {
            "food": "Late for my meal again. Pathetic.",
            "brushing": "You want to pet me? You have 3 seconds.",
            "isolation": "Silence. I'm sleeping. Finally.",
        },
    },
    "dog": {
        "excited_dog": {
            "food": "FOOD! FOOD! YES YES YES!",
            "bark": "PLAY WITH ME! Ball ball ball ball!",
            "growl": "LOOK AT ME! I'm HERE!",
            "grunt": "Ouch ouch ouch... that hurts...",
        },
        "shy_dog": {
            "food": "I'd love a little food, if you have time.",
            "bark": "I'd like to play with you, gently.",
            "growl": "I'm a bit scared. Will you protect me?",
            "grunt": "I don't feel well. Stay with me.",
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

    def translate(self, intent_category, personality="excited_dog", env_context=None, confidence=None, probabilities=None):
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

        return self._mock_response(intent_category, personality)

    def _mock_response(self, intent_category, personality):
        pet_key = "dog" if "dog" in personality else "cat"
        responses = MOCK_RESPONSES.get(pet_key, {})
        categories = responses.get(personality, {})
        if intent_category in categories:
            return categories[intent_category]
        for cat in categories.values():
            return cat
        return "Woof."
