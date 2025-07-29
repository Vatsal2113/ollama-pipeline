import sys
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch # Import torch for tensor operations

# Parse CLI arguments
model_id = sys.argv[1]
jsonl_path = sys.argv[2]

print(f"[INFO] Model: {model_id}")
print(f"[INFO] Dataset: {jsonl_path}")

# Load tokenizer with safe fallback
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    # Add a pad token if it doesn't exist, which is common for some models
    # It's often recommended to set pad_token to eos_token for causal LMs
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Fallback if no eos_token is available
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
except Exception as e:
    print(f"[ERROR] Failed to load tokenizer for {model_id}: {e}")
    sys.exit(1)

# Load model
try:
    model = AutoModelForCausalLM.from_pretrained(model_id)
    # If pad token was added after model load, resize embeddings here
    if tokenizer.pad_token is not None and model.get_input_embeddings().num_embeddings < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    
    # Ensure the model's pad_token_id is set for proper attention masking
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

except Exception as e:
    print(f"[ERROR] Failed to load model for {model_id}: {e}")
    sys.exit(1)


# Load dataset (local or S3)
try:
    dataset = load_dataset("json", data_files=jsonl_path, split="train")
except Exception as e:
    print(f"[ERROR] Failed to load dataset from {jsonl_path}: {e}")
    sys.exit(1)

# Format prompt: map 'question' to 'instruction' and 'answer' to 'output'
# Your train.jsonl has 'question' and 'answer' keys.
def format_prompt(example):
    instruction = example.get('question', '')
    response = example.get('answer', '')
    
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"


# Tokenize dataset
def tokenize(example):
    prompt = format_prompt(example)
    # Tokenize the prompt
    # return_tensors="pt" is handled by DataCollator, so remove it here
    tokenized_output = tokenizer(prompt, padding="max_length", truncation=True, max_length=512)
    
    # For causal language modeling, labels are typically the input_ids themselves.
    # The DataCollatorForLanguageModeling will handle the shifting of labels internally for loss calculation.
    # We don't need to clone input_ids to labels here if using DataCollatorForLanguageModeling
    # as it expects input_ids and creates labels from them.
    return tokenized_output

# Apply LoRA
print("[INFO] Applying LoRA configuration...")
lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05)
model = get_peft_model(model, lora_config)

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize, batched=False)

# Initialize DataCollator for Language Modeling
# This collator will handle padding and creating labels for causal LMs
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


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
    logging_steps=10,
    # Add a data collator if needed, but the default should work for simple cases with labels
    # data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator, # Pass the data collator here
)

# Start training
print("[INFO] Starting training...")
trainer.train()

# Save fine-tuned model and tokenizer
finetuned_path = os.path.join(output_dir, "finetuned")
model.save_pretrained(finetuned_path)
tokenizer.save_pretrained(finetuned_path)

print(f"[✓] Fine-tuned model saved to {finetuned_path}")
