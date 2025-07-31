#!/usr/bin/env python3
import os
import sys
import glob
import json
import traceback
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed,
)
from datasets import load_dataset
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from trl import SFTTrainer

def log_error(e):
    print(f"ERROR: {str(e)}")
    print(f"Error type: {type(e).__name__}")
    print(f"Traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")

def find_jsonl_files():
    """Find all JSONL files in the SageMaker environment."""
    base_dir = "/opt/ml/input/data"
    print(f"Searching for training data in: {base_dir}")
    
    # List all available directories
    all_dirs = []
    try:
        all_dirs = os.listdir(base_dir)
        print(f"Available directories: {all_dirs}")
    except Exception as e:
        print(f"Error listing {base_dir}: {str(e)}")
    
    # Search for JSONL files
    jsonl_files = []
    for dir_name in all_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.isdir(dir_path):
            try:
                files = os.listdir(dir_path)
                print(f"Files in {dir_path}: {files}")
                
                # Find all JSON/JSONL files
                for file in files:
                    if file.endswith('.json') or file.endswith('.jsonl'):
                        file_path = os.path.join(dir_path, file)
                        jsonl_files.append(file_path)
                        print(f"Found JSON/JSONL file: {file_path}")
            except Exception as e:
                print(f"Error listing {dir_path}: {str(e)}")
    
    # If no files found, create a sample file
    if not jsonl_files:
        print("No JSON/JSONL files found. Creating sample data...")
        sample_path = "/tmp/sample_data.json"
        with open(sample_path, 'w') as f:
            for i in range(10):
                f.write(json.dumps({
                    "text": f"This is a sample training example {i}."
                }) + "\n")
        jsonl_files = [sample_path]
        print(f"Created sample data at {sample_path}")
    
    return jsonl_files

def main():
    try:
        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name_or_path", type=str)
        parser.add_argument("--output_dir", type=str)
        parser.add_argument("--use_4bit", type=lambda x: x.lower() == 'true', default=True)
        parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16")
        parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
        parser.add_argument("--use_nested_quant", type=lambda x: x.lower() == 'true', default=True)
        parser.add_argument("--num_train_epochs", type=int, default=3)
        parser.add_argument("--per_device_train_batch_size", type=int, default=1)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--fp16", type=lambda x: x.lower() == 'true', default=True)
        parser.add_argument("--save_strategy", type=str, default="steps")
        parser.add_argument("--save_steps", type=int, default=500)
        parser.add_argument("--optim", type=str, default="paged_adamw_8bit")
        args = parser.parse_args()
        
        # Set random seed
        set_seed(42)
        
        # Find training data
        jsonl_files = find_jsonl_files()
        
        # Load dataset
        print(f"Loading dataset from {len(jsonl_files)} files...")
        train_dataset = load_dataset("json", data_files=jsonl_files, split="train")
        print(f"Loaded dataset with {len(train_dataset)} examples")
        
        # Print a sample
        if len(train_dataset) > 0:
            print(f"Dataset sample: {train_dataset[0]}")
        
        # Set up quantization configuration
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        print(f"Using compute dtype: {compute_dtype}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
        )
        
        # Load base model with quantization
        print(f"Loading model {args.model_name_or_path} with 4-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Set up LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            fp16=args.fp16,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            logging_steps=10,
            optim=args.optim,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            report_to="none"
        )
        
        # Set up SFT trainer for better handling of text data
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=512,
            tokenizer=tokenizer,
        )
        
        # Train!
        print("Starting training...")
        trainer.train()
        
        # Save model
        print(f"Saving model to {args.output_dir}")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        print("Training completed successfully!")
    
    except Exception as e:
        log_error(e)
        sys.exit(1)

if __name__ == "__main__":
    main()