# Ollama Pipeline

An automated pipeline for fine-tuning language models using LoRA and deploying them with Ollama.

## Overview

This pipeline automates the process of:
1. Fine-tuning language models using LoRA (Low-Rank Adaptation)
2. Converting models to GGUF format
3. Generating Ollama modelfiles
4. Uploading artifacts to S3

## Structure

```
├── Dockerfile                    # Docker image for SageMaker training
├── .github/workflows/ci-cd.yml  # GitHub Actions workflow
├── requirements.txt              # Python dependencies
├── scripts/
│   ├── pipeline.sh               # Main pipeline script
│   ├── generate_modelfile.py     # Enhanced modelfile generation
│   └── s3_upload.py             # Enhanced S3 upload utility
├── src/
│   ├── finetune_lora.py         # LoRA fine-tuning script
│   └── convert_to_gguf.py       # GGUF conversion script
└── data/
    └── train.jsonl              # Training data
```

## Features

### Enhanced Modelfile Generation
- Configurable parameters (temperature, top_k, top_p, etc.)
- Custom system prompts
- JSON configuration file support
- Proper error handling

### Robust S3 Upload
- Retry logic with exponential backoff
- Progress tracking
- Automatic bucket creation
- Content type detection

### GitHub Actions Integration
- Automated Docker builds
- Multi-model fine-tuning support
- Proper AWS credentials handling
- Error handling and validation

## Usage

### GitHub Actions Workflow

The pipeline automatically triggers on:
- Push to main branch
- Pull requests to main branch
- Manual workflow dispatch

Required GitHub Secrets:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

### Manual Pipeline Execution

```bash
# Run the pipeline for a specific model
scripts/pipeline.sh "model-name" "data/train.jsonl"
```

### Generate Modelfile

```bash
# Basic usage
scripts/generate_modelfile.py "model-name" "model.gguf" "output/Modelfile"

# With custom parameters
scripts/generate_modelfile.py "model-name" "model.gguf" "output/Modelfile" \
    --temperature 0.8 \
    --system-prompt "Custom system prompt" \
    --top-k 50
```

### Upload to S3

```bash
# Upload a file
scripts/s3_upload.py "local-file.txt" "my-bucket" "path/to/file.txt"

# Upload a directory
scripts/s3_upload.py "local-dir/" "my-bucket" "path/to/dir/" \
    --exclude "*.tmp" "*.log"
```

## Configuration

### Environment Variables

- `AWS_REGION`: AWS region (default: us-east-1)
- `ECR_REPO_NAME`: ECR repository name
- `S3_DATA_BUCKET`: S3 bucket for artifacts

### Model Configuration

Edit `scripts/pipeline.sh` to configure:
- AWS account ID
- SageMaker execution role
- Instance types
- S3 bucket names

## Testing

Run the validation test:

```bash
./test_pipeline.sh
```

This validates:
- File structure and permissions
- Script syntax
- Modelfile generation
- GitHub Actions workflow structure

## Recent Improvements

- ✅ Fixed Dockerfile location for better accessibility
- ✅ Enhanced modelfile generation with configurable parameters
- ✅ Added robust S3 upload with retry logic
- ✅ Improved GitHub Actions workflow with better error handling
- ✅ Added comprehensive validation testing
- ✅ Fixed undefined variables in pipeline script

## Requirements

- Python 3.10+
- Docker
- AWS CLI configured
- GitHub Actions (for automated workflow)

## Models Supported

The pipeline is configured to work with:
- `openlm-research/open_llama_3b`
- `distilbert-base-uncased`
- Any Hugging Face model compatible with Unsloth

## Architecture

1. **GitHub Actions** triggers the workflow
2. **Docker** builds the training environment
3. **SageMaker** runs the fine-tuning job
4. **Enhanced scripts** generate modelfiles and upload to S3
5. **Ollama** can use the generated modelfiles for deployment