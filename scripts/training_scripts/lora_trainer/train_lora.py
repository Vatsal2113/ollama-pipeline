#!/usr/bin/env python3

import os
import argparse
import logging
import sys
import traceback
import glob
import torch

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
    
    # Look for JSONL files in the training directory
    jsonl_files = glob.glob(os.path.join(base_dir, "**", "*.jsonl"), recursive=True)
    
    if jsonl_files:
        logger.info(f"Found {len(jsonl_files)} JSONL files: {jsonl_files}")
        return jsonl_files
    else:
        logger.warning("No JSONL files found in the training directory.")
        # Create a fallback sample file if no files found
        sample_path = os.path.join(base_dir, "fallback_sample.jsonl")
        with open(sample_path, "w") as f:
            f.write('{"text": "This is a fallback sample text for training."}\n')
        logger.info(f"Created fallback sample file: {sample_path}")
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
        
        # Parse arguments
        parser = argparse.ArgumentParser(description="Train a language model")
        parser.add_argument("--model_name_or_path", type=str, default="distilgpt2")
        parser.add_argument("--output_dir", type=str, default="/opt/ml/model")
        parser.add_argument("--per_device_train_batch_size", type=int, default=1)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
        parser.add_argument("--learning_rate", type=float, default=2e-4)
        parser.add_argument("--num_train_epochs", type=int, default=1)
        parser.add_argument("--use_lora", type=str, default='False')
        parser.add_argument("--lora_r", type=int, default=8)
        parser.add_argument("--lora_alpha", type=int, default=16)
        parser.add_argument("--lora_dropout", type=float, default=0.05)
        
        args = parser.parse_args()
        
        # Log all arguments
        logger.info(f"Arguments: {args}")
        
        # Find training files
        training_files = find_training_files()
        logger.info(f"Using training files: {training_files}")
        
        # Use the model specified in arguments
        model_name = args.model_name_or_path
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Custom dataset implementation to avoid PyArrow issues
        from torch.utils.data import Dataset
        
        class JsonlDataset(Dataset):
            def __init__(self, file_paths, tokenizer, max_length=64):
                self.examples = []
                
                for file_path in file_paths:
                    logger.info(f"Loading data from {file_path}")
                    try:
                        with open(file_path, 'r') as f:
                            for line in f:
                                try:
                                    # Handle different JSONL formats
                                    import json
                                    data = json.loads(line.strip())
                                    # Check if 'text' field exists, if not try to find text content
                                    if 'text' in data:
                                        text = data['text']
                                    elif 'content' in data:
                                        text = data['content']
                                    else:
                                        # Use the first string field we find
                                        for key, value in data.items():
                                            if isinstance(value, str) and len(value) > 10:
                                                text = value
                                                break
                                        else:
                                            continue  # Skip this example if no suitable text found
                                    
                                    encodings = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
                                    example = {key: torch.tensor(val) for key, val in encodings.items()}
                                    self.examples.append(example)
                                except Exception as e:
                                    logger.warning(f"Error processing line: {e}")
                                    continue
                    except Exception as e:
                        logger.error(f"Error loading file {file_path}: {e}")
                
                logger.info(f"Loaded {len(self.examples)} examples from {len(file_paths)} files")
                
                if not self.examples:
                    logger.warning("No examples loaded, creating one dummy example")
                    dummy_text = "This is a dummy example because no valid data was found."
                    encodings = tokenizer(dummy_text, truncation=True, max_length=max_length, padding="max_length")
                    example = {key: torch.tensor(val) for key, val in encodings.items()}
                    self.examples.append(example)
            
            def __len__(self):
                return len(self.examples)
            
            def __getitem__(self, idx):
                return self.examples[idx]
        
        # Load dataset
        train_dataset = JsonlDataset(training_files, tokenizer)
        logger.info(f"Dataset loaded with {len(train_dataset)} examples")
        
        # Load model
        logger.info(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        logger.info(f"Model loaded successfully")
        
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
            train_dataset=train_dataset,
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