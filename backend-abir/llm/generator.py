import os
from .rag_retriever import RAGRetriever
from .prompt import get_system_prompt, build_user_prompt
from llama_cpp import Llama

# Path where the GGUF model will be exported by Abir's convert.py
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "unsloth.Q4_K_M.gguf")

class PetTranslatorLLM:
    def __init__(self):
        self.rag = RAGRetriever()
        self.llm = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            print(f"Warning: Model not found at {MODEL_PATH}. Inference will fail.")
            return
            
        print("Loading GGUF Llama model...")
        self.llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_gpu_layers=-1 # Use GPU if available
        )
        print("Model loaded successfully.")

    def translate(self, intent_category, personality="excited_dog", env_context=None):
        """
        Translates an intent (provided by Anas's classification model) into a natural sentence.
        Uses RAG for context, env_context for environment variables, and Llama 3.2 for generation.
        """
        # 1. Retrieve behavioral context based on the intent (e.g. "dog showing Fear")
        query = f"behavior characteristic of a pet showing {intent_category}"
        rag_context = self.rag.retrieve_context(query)
        
        # 2. Build prompts
        system_prompt = get_system_prompt(personality)
        user_prompt = build_user_prompt(intent_category, rag_context, env_context)
        
        if self.llm is None:
            return f"[Simulated Translation for {intent_category}] - Model not loaded.\n\n--- Prompt that would be sent ---\n{user_prompt}"

        # 3. Generate response
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=60, # Keep sentences short and punchy
            temperature=0.7,
        )
        
        translation = response["choices"][0]["message"]["content"]
        return translation

if __name__ == "__main__":
    # Test integration without loading the actual heavy model
    translator = PetTranslatorLLM()
    # Mock input coming from Anas's TFJS/MobileNet output
    mock_intent_from_anas = "Fear" 
    mock_env = {
        "location": "outdoor",
        "weather": "thunderstorm",
        "time_of_day": "night",
        "other_animals": "none"
    }
    result = translator.translate(mock_intent_from_anas, personality="shy_dog", env_context=mock_env)
    print(f"\nRaw Intent: {mock_intent_from_anas}")
    print(f"Translation:\n{result}")
