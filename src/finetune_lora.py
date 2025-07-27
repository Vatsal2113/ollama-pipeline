import sys
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

model_id = sys.argv[1]
jsonl_path = sys.argv[2]

print(f"[INFO] Model: {model_id}")
print(f"[INFO] Dataset: {jsonl_path}")

# Load tokenizer safely
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
except Exception as e:
    print(f"[ERROR] Failed to load tokenizer: {e}")
    sys.exit(1)

# Load model safely
try:
    model = AutoModelForCausalLM.from_pretrained(model_id)
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    sys.exit(1)

# Load dataset (handles s3:// or local path)
try:
    dataset = load_dataset("json", data_files=jsonl_path, split="train")
except Exception as e:
    print(f"[ERROR] Failed to load dataset: {e}")
    sys.exit(1)

# Prompt formatting
def format_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"

def tokenize(example):
    prompt = format_prompt(example)
    return tokenizer(prompt, padding="max_length", truncation=True, max_length=512)

# Apply LoRA
lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05)
model = get_peft_model(model, lora_config)

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize)

# Training arguments
output_dir = f"./artifacts/{model_id.replace('/', '_')}"
args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    save_steps=20,
    save_total_limit=1,
    logging_dir='./logs'
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset
)

# Start training
trainer.train()

# Save model + tokenizer
finetune_path = os.path.join(output_dir, "finetuned")
model.save_pretrained(finetune_path)
tokenizer.save_pretrained(finetune_path)

print(f"[INFO] Fine-tuning completed. Model saved to {finetune_path}")
