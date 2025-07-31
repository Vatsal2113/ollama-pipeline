#!/usr/bin/env python3

import os
import argparse
import logging
import sys
import torch
from datasets import load_dataset
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
    TaskType,
    PeftConfig,
    PeftModel
)

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
        # Print version info first to help debug
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
        import peft
        logger.info(f"PEFT version: {peft.__version__}")
        
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
        
        logger.info(f"Training model: {args.model_name_or_path}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
        # Find training files
        training_files = find_training_files()
        logger.info(f"Using training files: {training_files}")
        
        # Load dataset
        dataset = load_dataset("json", data_files=training_files, split="train")
        logger.info(f"Loaded dataset with {len(dataset)} examples")
        logger.info(f"Sample: {dataset[0] if len(dataset) > 0 else 'No data'}")
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {args.model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenize function
        def tokenize(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
        
        # Process dataset
        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(tokenize, batched=True)
        logger.info(f"Dataset tokenized")
        
        # Load model
        logger.info(f"Loading model: {args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, 
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        logger.info(f"Model loaded successfully")
        
        # Set up LoRA if enabled
        if args.use_lora.lower() == 'true':
            logger.info("Setting up LoRA")
            # Get model architecture to configure proper target modules
            target_modules = []
            if "llama" in args.model_name_or_path.lower() or "mistral" in args.model_name_or_path.lower() or "tinyllama" in args.model_name_or_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            else:
                # Generic fallback
                target_modules = ["query_key_value"]
                
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
            model.print_trainable_parameters()
        
        # Set up data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            save_strategy="epoch",
            bf16=torch.cuda.is_available(),  # Use bfloat16 if available
            logging_steps=10,
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
        
        # Save model
        logger.info(f"Saving model to {args.output_dir}")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()