import torch
import os
import sys
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# SageMaker provides input data in /opt/ml/input/data/<channel_name>/
# We will use a 'training' channel
SM_CHANNEL_TRAINING = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
# SageMaker expects model artifacts to be saved to /opt/ml/model/
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

# Parse CLI arguments (these will be passed as hyperparameters to SageMaker)
# We'll expect model_id and jsonl_filename as hyperparameters
model_id = os.environ.get("SM_HP_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
jsonl_filename = os.environ.get("SM_HP_JSONL_FILENAME", "train.jsonl")

jsonl_path = os.path.join(SM_CHANNEL_TRAINING, jsonl_filename)

print(f"[INFO] Running on SageMaker. Model: {model_id}")
print(f"[INFO] Dataset path: {jsonl_path}")
print(f"[INFO] Model output directory: {SM_MODEL_DIR}")

# Configuration for Unsloth
max_seq_length = 256
dtype = None # Auto-detects based on GPU availability (bf16, fp16, or fp32)
load_in_4bit = True # Enable 4-bit quantization for memory savings

# Ensure the model output directory exists
os.makedirs(SM_MODEL_DIR, exist_ok=True)

# Check if dataset file exists
if not os.path.exists(jsonl_path):
    print(f"\nERROR: Dataset file '{jsonl_path}' not found at SageMaker input path!")
    sys.exit(1)

print("\nLoading base model and tokenizer with Unsloth...")
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_id,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    print("Model and tokenizer loaded.")
except Exception as e:
    print(f"[ERROR] Failed to load model or tokenizer for {model_id}: {e}")
    sys.exit(1)

print("\nConfiguring LoRA adapters with Unsloth...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
print("LoRA adapters configured.")

print(f"\nLoading dataset from '{jsonl_path}'...")
try:
    raw_dataset = load_dataset("json", data_files=jsonl_path, split="train")
    print(f"Dataset loaded. Number of examples: {len(raw_dataset)}")
except Exception as e:
    print(f"[ERROR] Failed to load dataset from {jsonl_path}: {e}")
    sys.exit(1)

def formatting_prompts_func(examples):
    questions = examples["question"]
    answers = examples["answer"]
    texts = []
    for q, a in zip(questions, answers):
        text = f"### Instruction:\n{q}\n\n### Response:\n{a}{tokenizer.eos_token}"
        texts.append(text)
    return {"text": texts}

dataset = raw_dataset.map(formatting_prompts_func, batched=True)
print(f"First formatted example for training:\n---\n{dataset[0]['text']}\n---")

print("\nDefining training arguments...")
training_args = TrainingArguments(
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 2,
    warmup_steps = 5,
    max_steps = 60,
    learning_rate = 2e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,
    output_dir = SM_MODEL_DIR, # SageMaker output directory for checkpoints
    optim = "paged_adamw_8bit",
    seed = 3407,
    save_strategy = "no", # We'll save GGUF directly via unsloth
    report_to = "none",
)
print("Training arguments defined.")

print("\nStarting fine-tuning...")
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    tokenizer = tokenizer,
    max_seq_length = max_seq_length,
    dataset_text_field = "text",
    args = training_args,
)

try:
    trainer.train()
    print("Fine-tuning complete.")
except Exception as e:
    print(f"[ERROR] Fine-tuning failed: {e}")
    sys.exit(1)

print(f"\nSaving fine-tuned model to {SM_MODEL_DIR} in GGUF format...")
# Unsloth saves the GGUF file directly into the specified directory
# The filename will be automatically generated (e.g., model-q4_k_m.gguf)
model.save_pretrained_gguf(SM_MODEL_DIR, tokenizer = tokenizer, quantization_method = "q4_k_m")

print(f"[✓] Fine-tuned GGUF model saved to: {SM_MODEL_DIR}/")

# SageMaker will automatically upload contents of SM_MODEL_DIR to S3
