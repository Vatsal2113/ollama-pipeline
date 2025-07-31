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
    
    if not os.path.exists(base_dir):
        logger.error(f"Directory not found: {base_dir}")
        # Create sample data as fallback
        logger.info("Creating sample training data")
        os.makedirs("/tmp/sample_data", exist_ok=True)
        with open("/tmp/sample_data/sample.jsonl", "w") as f:
            f.write('{"text": "This is a sample text for training."}\n')
            f.write('{"text": "Here is another example of training data."}\n')
        return ["/tmp/sample_data/sample.jsonl"]

    files = []
    for filename in os.listdir(base_dir):
        if filename.endswith('.json') or filename.endswith('.jsonl'):
            files.append(os.path.join(base_dir, filename))
            logger.info(f"Found training file: {filename}")
    
    if not files:
        logger.warning("No JSON/JSONL files found in training directory")
        # Create sample data
        sample_path = os.path.join(base_dir, "sample.jsonl")
        with open(sample_path, "w") as f:
            f.write('{"text": "This is a sample text for training."}\n')
            f.write('{"text": "Here is another example of training data."}\n')
        files.append(sample_path)
        logger.info(f"Created sample data at: {sample_path}")
    
    return files

def main():
    try:
        # Print environment information first
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9} GB")
        
        # Import libraries after logging versions
        try:
            import transformers
            logger.info(f"Transformers version: {transformers.__version__}")
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                Trainer,
                TrainingArguments,
                DataCollatorForLanguageModeling
            )
        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            raise
            
        try:
            import peft
            logger.info(f"PEFT version: {peft.__version__}")
            from peft import (
                get_peft_model,
                LoraConfig,
                TaskType
            )
        except ImportError as e:
            logger.error(f"Failed to import peft: {e}")
            raise
            
        try:
            import bitsandbytes as bnb
            logger.info(f"BitsAndBytes version: {bnb.__version__}")
        except ImportError:
            logger.warning("BitsAndBytes not available, will not use 8-bit optimization")
            bnb = None
        
        # Parse arguments
        parser = argparse.ArgumentParser(description="Train a language model with LoRA")
        
        # Required parameters from SageMaker
        parser.add_argument("--model_name_or_path", type=str, required=True)
        parser.add_argument("--output_dir", type=str, required=True)
        
        # Optional parameters
        parser.add_argument("--per_device_train_batch_size", type=int, default=2)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        parser.add_argument("--learning_rate", type=float, default=2e-4)
        parser.add_argument("--num_train_epochs", type=int, default=1)
        
        # LoRA parameters
        parser.add_argument("--use_lora", type=str, default='True')
        parser.add_argument("--lora_r", type=int, default=8)
        parser.add_argument("--lora_alpha", type=int, default=16)
        parser.add_argument("--lora_dropout", type=float, default=0.05)
        
        args = parser.parse_args()
        
        # Log all arguments
        logger.info(f"Arguments: {args}")
        logger.info(f"Training model: {args.model_name_or_path}")
        
        # Find training files
        training_files = find_training_files()
        logger.info(f"Using training files: {training_files}")
        
        # Load dataset
        try:
            dataset = load_dataset("json", data_files=training_files, split="train")
            logger.info(f"Loaded dataset with {len(dataset)} examples")
            if len(dataset) > 0:
                logger.info(f"Sample: {dataset[0]}")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
        
        # Load tokenizer
        try:
            logger.info(f"Loading tokenizer: {args.model_name_or_path}")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
        
        # Tokenize function
        def tokenize(examples):
            return tokenizer(
                examples["text"], 
                truncation=True, 
                padding="max_length", 
                max_length=128
            )
        
        # Process dataset
        try:
            logger.info("Tokenizing dataset...")
            tokenized_dataset = dataset.map(tokenize, batched=True)
            logger.info(f"Dataset tokenized successfully")
        except Exception as e:
            logger.error(f"Error tokenizing dataset: {e}")
            raise
        
        # Load model - Use 8-bit quantization to reduce memory footprint
        try:
            logger.info(f"Loading model: {args.model_name_or_path}")
            
            # Check if bitsandbytes is available
            load_in_8bit = False
            if bnb is not None:
                load_in_8bit = True
                logger.info("Using 8-bit quantization")
            
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                load_in_8bit=load_in_8bit,
                device_map="auto" if torch.cuda.is_available() else None
            )
            logger.info(f"Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(traceback.format_exc())
            raise
        
        # Set up LoRA if enabled
        if args.use_lora.lower() == 'true':
            try:
                logger.info("Setting up LoRA")
                
                # Simple target modules that work well for most models
                target_modules = ["q_proj", "v_proj"]
                
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
                logger.info("LoRA applied to model successfully")
            except Exception as e:
                logger.error(f"Error setting up LoRA: {e}")
                logger.error(traceback.format_exc())
                raise
        
        # Set up data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False
        )
        
        # Training arguments
        try:
            training_args = TrainingArguments(
                output_dir=args.output_dir,
                per_device_train_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=args.learning_rate,
                num_train_epochs=args.num_train_epochs,
                save_strategy="epoch",
                logging_steps=10,
                remove_unused_columns=False
            )
        except Exception as e:
            logger.error(f"Error setting up training arguments: {e}")
            raise
        
        # Initialize trainer
        try:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer
            )
        except Exception as e:
            logger.error(f"Error initializing trainer: {e}")
            raise
        
        # Train model
        try:
            logger.info("Starting training...")
            trainer.train()
            logger.info("Training completed successfully!")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            logger.error(traceback.format_exc())
            raise
        
        # Save model
        try:
            logger.info(f"Saving model to {args.output_dir}")
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            logger.info(f"Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
        
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()