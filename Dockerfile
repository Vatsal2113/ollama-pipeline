FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir boto3 sagemaker pandas

# Copy necessary files
COPY pipeline.sh /app/
COPY sagemaker_finetune.py /app/
COPY scripts /app/scripts/

# Make scripts executable
RUN chmod +x /app/pipeline.sh

# Entry point
CMD ["/app/pipeline.sh"]