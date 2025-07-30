#!/bin/bash
set -e

MODEL_NAME="$1"
JSONL_PATH="$2" # This is the local path to the JSONL file
JSONL_FILENAME=$(basename "$JSONL_PATH") # Extract just the filename

AWS_REGION="us-east-1" # Set your desired AWS region
AWS_ACCOUNT_ID="504212701662" # <<< UPDATED with your AWS Account ID
ECR_REPO_NAME="ollama-lora-sagemaker"
SAGEMAKER_ROLE_ARN="arn:aws:iam::504212701662:role/MySageMakerExecutionRoleForOllama" # <<< UPDATED with your SageMaker Execution Role ARN
SAGEMAKER_INSTANCE_TYPE="ml.g4dn.xlarge" # Recommended for GPU training. Adjust as needed.
SAGEMAKER_INSTANCE_COUNT=1
SAGEMAKER_JOB_NAME="ollama-lora-finetune-${MODEL_NAME//\//_}-$(date +%s)"
S3_DATA_BUCKET="ollama-lora-pipeline" # Your existing S3 bucket for data/artifacts
S3_DATA_PREFIX="sagemaker-input-data/${MODEL_NAME//\//_}" # Prefix for input data
S3_OUTPUT_PREFIX="sagemaker-output-models/${MODEL_NAME//\//_}" # Prefix for SageMaker job output

echo "[*] Fine-tuning model: $MODEL_NAME"
echo "[*] Dataset: $JSONL_PATH"

# --- 1. Login to ECR and build/push Docker image ---
echo "[*] Logging in to ECR..."
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID".dkr.ecr."$AWS_REGION".amazonaws.com

echo "[*] Building Docker image for SageMaker..."
docker build -t "$ECR_REPO_NAME" -f Dockerfile .

echo "[*] Tagging Docker image..."
docker tag "$ECR_REPO_NAME" "$AWS_ACCOUNT_ID".dkr.ecr."$AWS_REGION".amazonaws.com/"$ECR_REPO_NAME":latest

echo "[*] Pushing Docker image to ECR..."
aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$AWS_REGION" || true # Create repo if it doesn't exist
docker push "$AWS_ACCOUNT_ID".dkr.ecr."$AWS_REGION".amazonaws.com/"$ECR_REPO_NAME":latest
ECR_IMAGE_URI="$AWS_ACCOUNT_ID".dkr.ecr."$AWS_REGION".amazonaws.com/"$ECR_REPO_NAME":latest
echo "[✓] Docker image pushed to ECR: $ECR_IMAGE_URI"

# --- 2. Upload dataset to S3 for SageMaker input ---
echo "[*] Uploading dataset to S3: s3://$S3_DATA_BUCKET/$S3_DATA_PREFIX/$JSONL_FILENAME"
aws s3 cp "$JSONL_PATH" "s3://$S3_DATA_BUCKET/$S3_DATA_PREFIX/$JSONL_FILENAME"
S3_INPUT_DATA_URI="s3://$S3_DATA_BUCKET/$S3_DATA_PREFIX"
echo "[✓] Dataset uploaded to S3."

# --- 3. Create SageMaker Training Job ---
echo "[*] Creating SageMaker training job: $SAGEMAKER_JOB_NAME"

aws sagemaker create-training-job \
    --training-job-name "$SAGEMAKER_JOB_NAME" \
    --hyperparameters "{\"model_id\":\"$MODEL_NAME\",\"jsonl_filename\":\"$JSONL_FILENAME\"}" \
    --algorithm-specification "TrainingImage=$ECR_IMAGE_URI,TrainingInputMode=File" \
    --role-arn "$SAGEMAKER_ROLE_ARN" \
    --input-data-config "[{ \"ChannelName\": \"training\", \"DataSource\": { \"S3DataSource\": { \"S3DataType\": \"S3Prefix\", \"S3Uri\": \"$S3_INPUT_DATA_URI\", \"S3DataDistributionType\": \"FullyReplicated\" } } }]" \
    --output-data-config "S3OutputPath=s3://$S3_DATA_BUCKET/$S3_OUTPUT_PREFIX/" \
    --resource-config "InstanceType=$SAGEMAKER_INSTANCE_TYPE,InstanceCount=$SAGEMAKER_INSTANCE_COUNT,VolumeSizeInGB=50" \
    --stopping-condition "MaxRuntimeInSeconds=3600" \
    --region "$AWS_REGION"

echo "[✓] SageMaker training job created. Monitoring progress..."

# --- 4. Monitor SageMaker Training Job ---
STATUS="InProgress"
while [ "$STATUS" == "InProgress" ] || [ "$STATUS" == "Starting" ] || [ "$STATUS" == "Downloading" ] || [ "$STATUS" == "Uploading" ]; do
    sleep 60 # Check every 60 seconds
    JOB_INFO=$(aws sagemaker describe-training-job --training-job-name "$SAGEMAKER_JOB_NAME" --region "$AWS_REGION")
    STATUS=$(echo "$JOB_INFO" | jq -r '.TrainingJobStatus')
    echo "[*] SageMaker job status: $STATUS"
    if [ "$STATUS" == "Failed" ] || [ "$STATUS" == "Stopped" ]; then
        echo "[ERROR] SageMaker training job failed or was stopped."
        echo "$JOB_INFO" | jq '.FailureReason'
        exit 1
    fi
done

if [ "$STATUS" == "Completed" ]; then
    echo "[✓] SageMaker training job completed successfully!"
    SAGEMAKER_MODEL_OUTPUT_URI=$(echo "$JOB_INFO" | jq -r '.ModelArtifacts.S3ModelArtifacts')
    echo "[*] SageMaker model artifacts saved to: $SAGEMAKER_MODEL_OUTPUT_URI"
else
    echo "[ERROR] Unexpected SageMaker job status: $STATUS"
    exit 1
fi

# --- 5. Copy GGUF from SageMaker output to your target bucket ---
echo "[*] Copying GGUF model from SageMaker output to target S3 bucket..."

# SageMaker typically saves model artifacts as a model.tar.gz file in the S3OutputPath.
# We need to download this, extract, find the GGUF, and then re-upload.
echo "[*] Downloading SageMaker model artifact tarball..."
mkdir -p "$FINETUNED_UNSLOTH_DIR" # Create a local directory to extract
aws s3 cp "$SAGEMAKER_MODEL_OUTPUT_URI" "$FINETUNED_UNSLOTH_DIR/model.tar.gz"
tar -xzf "$FINETUNED_UNSLOTH_DIR/model.tar.gz" -C "$FINETUNED_UNSLOTH_DIR"

# Now find the GGUF file within the extracted directory
GGUF_FILENAME=$(find "$FINETUNED_UNSLOTH_DIR" -maxdepth 2 -name "*.gguf" -print -quit) # Increased maxdepth to 2
if [ -z "$GGUF_FILENAME" ]; then
    echo "[ERROR] GGUF file not found after extracting SageMaker artifact."
    ls -alR "$FINETUNED_UNSLOTH_DIR" || true
    exit 1
fi
echo "[✓] Found GGUF file locally: $GGUF_FILENAME"

OLLAMA_MODEL_NAME="${MODEL_NAME//\//_}-finetuned"
TARGET_GGUF_S3_PATH="s3://$S3_DATA_BUCKET/gguf/${OLLAMA_MODEL_NAME}.gguf"
aws s3 cp "$GGUF_FILENAME" "$TARGET_GGUF_S3_PATH"
echo "[✓] GGUF model copied to: $TARGET_GGUF_S3_PATH"

# --- 6. Create and Upload Ollama Modelfile ---
echo "[*] Creating Ollama Modelfile..."
MODFILE_LOCAL_PATH="${FINETUNED_UNSLOTH_DIR}/Modelfile"
GGUF_BASENAME=$(basename "$GGUF_FILENAME") # Get just the filename for Modelfile

cat > "${MODFILE_LOCAL_PATH}" <<EOF
FROM ./${GGUF_BASENAME}
PARAMETER temperature 0.7
SYSTEM You are a helpful assistant.
EOF

TARGET_MODFILE_S3_PATH="s3://$S3_DATA_BUCKET/modelfiles/${OLLAMA_MODEL_NAME}.Modelfile"
aws s3 cp "${MODFILE_LOCAL_PATH}" "$TARGET_MODFILE_S3_PATH"
echo "[✓] Modelfile uploaded to: $TARGET_MODFILE_S3_PATH"

echo "[✓] Pipeline complete"
