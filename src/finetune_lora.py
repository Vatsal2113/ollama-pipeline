import sys
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Parse CLI arguments
model_id = sys.argv[1]
jsonl_path = sys.argv[2]

print(f"[INFO] Model: {model_id}")
print(f"[INFO] Dataset: {jsonl_path}")

# Load tokenizer with safe fallback
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
except Exception as e:
    print(f"[ERROR] Failed to load tokenizer for {model_id}: {e}")
    sys.exit(1)

# Load model
try:
    model = AutoModelForCausalLM.from_pretrained(model_id)
except Exception as e:
    print(f"[ERROR] Failed to load model for {model_id}: {e}")
    sys.exit(1)

# Load dataset (local or S3)
try:
    dataset = load_dataset("json", data_files=jsonl_path, split="train")
except Exception as e:
    print(f"[ERROR] Failed to load dataset from {jsonl_path}: {e}")
    sys.exit(1)

# Format prompt: instruction + input => output
def format_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"

# Tokenize dataset
def tokenize(example):
    prompt = format_prompt(example)
    return tokenizer(prompt, padding="max_length", truncation=True, max_length=512)

# Apply LoRA
print("[INFO] Applying LoRA configuration...")
lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05)
model = get_peft_model(model, lora_config)

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize)

# Training output path
output_dir = f"./artifacts/{model_id.replace('/', '_')}"

# Training setup
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    save_steps=20,
    save_total_limit=1,
    logging_dir='./logs',
    logging_steps=10
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# Start training
print("[INFO] Starting training...")
trainer.train()

# Save fine-tuned model and tokenizer
finetuned_path = os.path.join(output_dir, "finetuned")
model.save_pretrained(finetuned_path)
tokenizer.save_pretrained(finetuned_path)

print(f"[✓] Fine-tuned model saved to {finetuned_path}")
