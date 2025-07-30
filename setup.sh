#!/bin/bash

# Create directory for Ollama data
mkdir -p ollama-data

# Create directory for training scripts
mkdir -p scripts/training_scripts

# Create a simple training script
cat > scripts/training_scripts/train.py << 'EOF'
#!/usr/bin/env python3

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

def main():
    # Load model and tokenizer
    model_name = os.environ.get("model_name_or_path")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Configure LoRA
    if os.environ.get("USE_LORA", "false").lower() == "true":
        lora_r = int(os.environ.get("LORA_R", "16"))
        lora_alpha = int(os.environ.get("LORA_ALPHA", "32"))
        lora_dropout = float(os.environ.get("LORA_DROPOUT", "0.05"))
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Load dataset
    training_data_path = os.environ.get("TRAINING_DATA_PATH")
    dataset = load_dataset("json", data_files=f"{training_data_path}/*.jsonl")
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=os.environ.get("output_dir", "/opt/ml/model"),
        per_device_train_batch_size=int(os.environ.get("per_device_train_batch_size", "8")),
        gradient_accumulation_steps=int(os.environ.get("gradient_accumulation_steps", "4")),
        learning_rate=float(os.environ.get("learning_rate", "2e-4")),
        num_train_epochs=int(os.environ.get("num_train_epochs", "3")),
        fp16=os.environ.get("fp16", "true").lower() == "true",
        save_strategy="epoch",
    )
    
    # Setup trainer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model()

if __name__ == "__main__":
    main()
EOF