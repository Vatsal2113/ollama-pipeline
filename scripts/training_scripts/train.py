#!/usr/bin/env python3
"""
Training script for SageMaker fine-tuning
"""

import os
import argparse
import logging
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--output_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--use_lora", type=bool, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--load_in_8bit", type=bool, default=True)
    parser.add_argument("--use_flash_attention", type=bool, default=False)
    
    # Parse args
    args, _ = parser.parse_known_args()
    return args

def setup_model_and_tokenizer(args):
    logger.info(f"Loading model {args.model_name_or_path}...")
    
    # Load model with quantization if enabled
    if args.load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    
    # Apply LoRA if enabled
    if args.use_lora:
        logger.info("Applying LoRA configuration...")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"]
        )
        model = get_peft_model(model, peft_config)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def load_and_prepare_data(tokenizer):
    logger.info("Loading and preparing dataset...")
    # In SageMaker, data is automatically downloaded to /opt/ml/input/data/train
    data_path = "/opt/ml/input/data/train"
    
    # Load dataset
    dataset = load_dataset("json", data_files=os.path.join(data_path, "*.jsonl"))
    
    # Preprocess dataset
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["text"])
    
    return tokenized_dataset

def train(args, model, tokenizer, dataset):
    logger.info("Setting up training...")
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        dataloader_num_workers=4,
        group_by_length=True,
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training complete!")

def main():
    args = parse_args()
    model, tokenizer = setup_model_and_tokenizer(args)
    dataset = load_and_prepare_data(tokenizer)
    train(args, model, tokenizer, dataset)

if __name__ == "__main__":
    main()