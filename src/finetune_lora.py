import sys
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

model_id = sys.argv[1]
jsonl_path = sys.argv[2]  # Can be local or s3:// path

print(f"Model: {model_id}")
print(f"Dataset: {jsonl_path}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Load JSONL dataset
dataset = load_dataset("json", data_files=jsonl_path, split="train")

# Format prompt: instruction + input => output
def format_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"

def tokenize(example):
    prompt = format_prompt(example)
    return tokenizer(prompt, padding="max_length", truncation=True, max_length=512)

# Apply LoRA
lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05)
model = get_peft_model(model, lora_config)

# Tokenize
tokenized_dataset = dataset.map(tokenize)

# Training config
output_dir = f"./artifacts/{model_id.replace('/', '_')}"
args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    save_steps=20,
    save_total_limit=1,
    logging_dir='./logs'
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset
)

trainer.train()

# Save model
model.save_pretrained(os.path.join(output_dir, "finetuned"))
tokenizer.save_pretrained(os.path.join(output_dir, "finetuned"))
