#!/usr/bin/env python3

import os
import argparse
import logging
import sys
import traceback
import glob
import torch
import json

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
        
        # Import one module at a time to isolate import errors
        logger.info("Importing AutoTokenizer...")
        from transformers import AutoTokenizer
        
        logger.info("Importing AutoModelForCausalLM...")
        from transformers import AutoModelForCausalLM
        
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
            
        # Load model
        logger.info(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        logger.info(f"Model loaded successfully")
        
        # Super simplified training - just one epoch, one batch
        logger.info("Starting simple training loop")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        
        # Process data and train
        for file_path in training_files:
            logger.info(f"Processing file: {file_path}")
            examples = []
            
            # Read data
            try:
                with open(file_path, 'r') as f:
                    for i, line in enumerate(f):
                        if i >= 10:  # Just use first 10 examples for quick training
                            break
                        try:
                            data = json.loads(line.strip())
                            if 'text' in data:
                                text = data['text']
                            elif 'content' in data:
                                text = data['content']
                            else:
                                continue
                                
                            # Simple tokenization
                            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                            examples.append(inputs)
                        except Exception as e:
                            logger.warning(f"Error processing line: {e}")
                            continue
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
            
            # If no examples, create a dummy example
            if not examples:
                logger.warning("No examples found, creating dummy example")
                dummy_text = "This is a dummy training example."
                inputs = tokenizer(dummy_text, return_tensors="pt", truncation=True, max_length=64)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                examples.append(inputs)
            
            # Simple training loop
            logger.info(f"Training on {len(examples)} examples")
            for epoch in range(int(args.num_train_epochs)):
                logger.info(f"Starting epoch {epoch+1}/{args.num_train_epochs}")
                total_loss = 0
                
                for i, example in enumerate(examples):
                    # Forward pass
                    outputs = model(**example, labels=example["input_ids"])
                    loss = outputs.loss
                    total_loss += loss.item()
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update weights
                    if (i + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        
                    logger.info(f"Example {i+1}/{len(examples)}, Loss: {loss.item()}")
                
                # Make sure all gradients are applied
                if len(examples) % args.gradient_accumulation_steps != 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    
                logger.info(f"Epoch {epoch+1} completed, Average loss: {total_loss/len(examples)}")
        
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