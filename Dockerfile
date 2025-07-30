FROM python:3.9-slim

WORKDIR /app

# Install system dependencies including curl
RUN apt-get update && \
    apt-get install -y curl wget git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir boto3 sagemaker pandas transformers datasets peft

# Copy necessary files
COPY pipeline.sh /app/
COPY sagemaker_finetune.py /app/
COPY scripts /app/scripts/

# Make scripts executable
RUN chmod +x /app/pipeline.sh

# Entry point
CMD ["/app/pipeline.sh"]