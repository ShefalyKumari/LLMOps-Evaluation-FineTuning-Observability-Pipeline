import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "outputs/lora_adapter"
DATA_PATH = "data/dataset.json"

def prepare_dataset(tokenizer, data_path: str):
    """Loads and formats instruction dataset into tokenized prompt-completion pairs."""
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    formatted = []
    for item in raw_data:
        prompt = (
            f"<|im_start|>system\nYou are an enterprise support specialist. Answer strictly using context.<|im_end|>\n"
            f"<|im_start|>user\nContext: {item['context']}\nQuestion: {item['query']}<|im_end|>\n"
            f"<|im_start|>assistant\n{item['ground_truth']}<|im_end|>"
        )
        formatted.append({"text": prompt})

    dataset = Dataset.from_list(formatted)

    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding="max_length"
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    return dataset.map(tokenize_fn, batched=True)

def train_lora():
    print(f"1. Loading tokenizer and base model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Executing fine-tuning on compute device: {device.upper()}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        low_cpu_mem_usage=True
    )

    print("2. Configuring LoRA (Low-Rank Adaptation) adapter...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    print("3. Preprocessing domain dataset...")
    train_dataset = prepare_dataset(tokenizer, DATA_PATH)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=1,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset
    )

    print("4. Starting PEFT fine-tuning run...")
    trainer.train()

    print(f"5. Saving LoRA adapter checkpoints to '{OUTPUT_DIR}'...")
    peft_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Fine-tuning run complete. Adapter weights serialized successfully.")

if __name__ == "__main__":
    train_lora()