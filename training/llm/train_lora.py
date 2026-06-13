import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# Configuration
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"  # 1B model as requested in agents_ia.md
MAX_SEQ_LENGTH = 2048
DATASET_PATH = "synthetic_dataset.json"

# LoRA parameters
LORA_R = 16
LORA_ALPHA = 16

def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset {DATASET_PATH} not found. Please run dataset.py first.")

    print("Loading Llama 3.2 model via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None, # Auto detect
        load_in_4bit = True, # QLoRA
    )

    print("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = LORA_R,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = LORA_ALPHA,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    print("Loading dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    def format_prompts(examples):
        instructions = examples["instruction"]
        outputs = examples["output"]
        texts = []
        for inst, out in zip(instructions, outputs):
            # Format according to Llama 3 instruction format or simple ChatML
            text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{inst}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{out}<|eot_id|>"
            texts.append(text)
        return { "text" : texts }

    dataset = dataset.map(format_prompts, batched=True)

    print("Initializing trainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60, # Small number for demo/testing
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )

    print("Starting training...")
    trainer_stats = trainer.train()
    
    print("Saving LoRA adapters...")
    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")
    
    print("Training complete!")

if __name__ == "__main__":
    main()
