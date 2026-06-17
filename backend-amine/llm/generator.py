import os
import random
from .prompt import get_system_prompt, build_user_prompt

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "unsloth.Q4_K_M.gguf")

MOCK_RESPONSES = {
    "cat": {
        "haughty_cat": {
            "hunger": "Finally, you decide to feed your king.",
            "play": "That piece of string moves. Interesting.",
            "attention": "You may pet me, I grant you this favor.",
            "fear": "There is an intruder in my kingdom.",
            "pain": "Something is wrong in my domain.",
            "content": "You have my favor for today.",
        },
        "grumpy_cat": {
            "hunger": "Late for my meal again. Pathetic.",
            "play": "You want to play? You have 30 seconds.",
            "attention": "What now? Hurry up.",
            "fear": "There's a noise. Go check. Now.",
            "pain": "Something's wrong. Move it.",
            "content": "Silence. I'm sleeping. Finally.",
        },
    },
    "dog": {
        "excited_dog": {
            "hunger": "FOOD! FOOD! YES YES YES!",
            "play": "PLAY WITH ME! Ball ball ball ball!",
            "attention": "LOOK AT ME! I'm HERE!",
            "fear": "I'm scared... STAY WITH ME!",
            "pain": "Ouch ouch ouch... that hurts...",
            "content": "I love you. You're the best. Life is beautiful.",
        },
        "shy_dog": {
            "hunger": "I'd love a little food, if you have time.",
            "play": "I'd like to play with you, gently.",
            "attention": "I'm here, next to you. I love you.",
            "fear": "I'm a bit scared. Will you protect me?",
            "pain": "I don't feel well. Stay with me.",
            "content": "Everything is fine. I'm happy with you.",
        },
    },
}

class PetTranslatorLLM:
    def __init__(self):
        self.llm = None
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

    def translate(self, intent_category, personality="excited_dog", env_context=None):
        if self.llm is not None:
            try:
                system_prompt = get_system_prompt(personality)
                user_prompt = build_user_prompt(intent_category, None, env_context)
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
