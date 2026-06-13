from unsloth import FastLanguageModel
import os

MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
LORA_PATH = "lora_model"
EXPORT_PATH = "../../backend/models"

def main():
    if not os.path.exists(LORA_PATH):
        raise FileNotFoundError(f"LoRA adapters not found in {LORA_PATH}. Run train_lora.py first.")

    print("Loading model and LoRA adapters...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    
    # Load the trained LoRA
    model.load_adapter(LORA_PATH)

    os.makedirs(EXPORT_PATH, exist_ok=True)

    print("Exporting to GGUF format (Q4_K_M)...")
    # Save to q4_k_m GGUF format for llama.cpp
    model.save_pretrained_gguf(EXPORT_PATH, tokenizer, quantization_method = "q4_k_m")
    print(f"Export complete. GGUF file saved in {EXPORT_PATH}")

if __name__ == "__main__":
    main()
