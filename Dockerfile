FROM python:3.9

WORKDIR /app

# Install required packages
RUN apt-get update && apt-get install -y curl
RUN pip install --no-cache-dir awscli boto3 sagemaker

# Install additional requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p scripts/training_scripts

# Copy scripts
COPY pipeline.sh .
COPY extract_gguf.py .
COPY sagemaker_finetune.py .
COPY scripts/ scripts/

# Make scripts executable
RUN chmod +x pipeline.sh extract_gguf.py sagemaker_finetune.py
RUN chmod +x scripts/*.py scripts/training_scripts/*.py || true

CMD ["./pipeline.sh"]