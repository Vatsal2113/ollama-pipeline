FROM python:3.9-slim

WORKDIR /app

# Install system dependencies including curl and AWS CLI
RUN apt-get update && \
    apt-get install -y curl wget git unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Install AWS CLI v2
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf aws awscliv2.zip

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