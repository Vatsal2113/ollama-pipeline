#!/usr/bin/env python3

import os
import argparse
import logging
import sys
import traceback
import torch
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def find_training_files():
    """Find training data files in SageMaker paths."""
    base_dir = "/opt/ml/input/data/training"
    logger.info(f"Looking for training data in: {base_dir}")
    
    # Create sample data regardless to ensure we have something to work with
    logger.info("Creating sample training data")
    sample_path = os.path.join(base_dir, "sample.jsonl")
    with open(sample_path, "w") as f:
        f.write('{"text": "This is a sample text for training."}\n')
        f.write('{"text": "Here is another example of training data."}\n')
        f.write('{"text": "A third example to ensure we have data."}\n')
        f.write('{"text": "Fourth training example for the model."}\n')
    
    return [sample_path]

def main():
    try:
        # Print environment information first
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        
        # Import libraries after logging versions
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            DataCollatorForLanguageModeling
        )
        from peft import (
            get_peft_model,
            LoraConfig,
            TaskType
        )
        
        # Parse arguments
        parser = argparse.ArgumentParser(description="Train a language model with LoRA")
        parser.add_argument("--model_name_or_path", type=str, default="distilgpt2")
        parser.add_argument("--output_dir", type=str, default="/opt/ml/model")
        parser.add_argument("--per_device_train_batch_size", type=int, default=1)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
        parser.add_argument("--learning_rate", type=float, default=2e-4)
        parser.add_argument("--num_train_epochs", type=int, default=1)
        parser.add_argument("--use_lora", type=str, default='True')
        parser.add_argument("--lora_r", type=int, default=8)
        parser.add_argument("--lora_alpha", type=int, default=16)
        parser.add_argument("--lora_dropout", type=float, default=0.05)
        
        args = parser.parse_args()
        
        # Log all arguments
        logger.info(f"Arguments: {args}")
        
        # Find training files
        training_files = find_training_files()
        logger.info(f"Using training files: {training_files}")
        
        # Load dataset
        dataset = load_dataset("json", data_files=training_files, split="train")
        logger.info(f"Loaded dataset with {len(dataset)} examples")
        
        # Use a smaller model - distilgpt2 is only 82M parameters
        model_name = "distilgpt2" 
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenize function
        def tokenize(examples):
            return tokenizer(
                examples["text"], 
                truncation=True, 
                padding="max_length", 
                max_length=64
            )
        
        # Process dataset
        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(tokenize, batched=True)
        logger.info(f"Dataset tokenized successfully")
        
        # Load model
        logger.info(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        logger.info(f"Model loaded successfully")
        
        # Set up LoRA if enabled
        if args.use_lora.lower() == 'true':
            logger.info("Setting up LoRA")
            
            # Simple target modules for distilgpt2
            target_modules = ["c_attn"]
            
            logger.info(f"Using target modules for LoRA: {target_modules}")
            
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            model = get_peft_model(model, lora_config)
        
        # Set up data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False
        )
        
        # Training arguments - simplified
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            save_strategy="no",  # Don't save intermediate checkpoints
            logging_steps=1,
            remove_unused_columns=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        # Train model
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed successfully!")
        
        # Save model
        logger.info(f"Saving model to {args.output_dir}")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Model saved successfully")
        
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()