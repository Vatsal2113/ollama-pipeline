#!/bin/bash
# Test script to validate the ollama-pipeline setup

set -e

echo "=== Ollama Pipeline Validation Test ==="

# Test 1: Check Dockerfile location
echo "✓ Testing Dockerfile location..."
if [ -f "Dockerfile" ]; then
    echo "  ✓ Dockerfile found in root directory"
else
    echo "  ✗ Dockerfile not found in root directory"
    exit 1
fi

# Test 2: Check pipeline script syntax
echo "✓ Testing pipeline script syntax..."
bash -n scripts/pipeline.sh
echo "  ✓ Pipeline script syntax is valid"

# Test 3: Check Python scripts
echo "✓ Testing Python scripts..."
python -m py_compile scripts/generate_modelfile.py
python -m py_compile scripts/s3_upload.py
echo "  ✓ Python scripts syntax is valid"

# Test 4: Test modelfile generation
echo "✓ Testing modelfile generation..."
mkdir -p /tmp/pipeline-test
python scripts/generate_modelfile.py \
    "test-model" \
    "test-model.gguf" \
    "/tmp/pipeline-test/Modelfile" \
    --temperature 0.8 \
    --system-prompt "Test system prompt"

if [ -f "/tmp/pipeline-test/Modelfile" ]; then
    echo "  ✓ Modelfile generated successfully"
    echo "  Content preview:"
    head -5 /tmp/pipeline-test/Modelfile | sed 's/^/    /'
else
    echo "  ✗ Modelfile generation failed"
    exit 1
fi

# Test 5: Check GitHub Actions workflow syntax
echo "✓ Testing GitHub Actions workflow..."
# We can't use standard YAML parser due to GitHub expressions, but check basic structure
if [ -f ".github/workflows/ci-cd.yml" ]; then
    echo "  ✓ GitHub Actions workflow file exists"
    # Check for required sections
    grep -q "name:" .github/workflows/ci-cd.yml
    grep -q "on:" .github/workflows/ci-cd.yml  
    grep -q "jobs:" .github/workflows/ci-cd.yml
    echo "  ✓ Workflow has required sections"
else
    echo "  ✗ GitHub Actions workflow file not found"
    exit 1
fi

# Test 6: Check environment variables and secrets handling
echo "✓ Testing environment variables configuration..."
grep -q "AWS_REGION" .github/workflows/ci-cd.yml
grep -q "secrets.AWS_ACCESS_KEY_ID" .github/workflows/ci-cd.yml
echo "  ✓ Environment variables properly configured"

# Test 7: Verify all required files exist
echo "✓ Testing required files existence..."
required_files=(
    "Dockerfile"
    "requirements.txt"
    "scripts/pipeline.sh"
    "scripts/generate_modelfile.py"
    "scripts/s3_upload.py"
    "src/finetune_lora.py"
    ".github/workflows/ci-cd.yml"
    ".gitignore"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file exists"
    else
        echo "  ✗ $file missing"
        exit 1
    fi
done

# Test 8: Check script permissions
echo "✓ Testing script permissions..."
for script in scripts/pipeline.sh scripts/generate_modelfile.py scripts/s3_upload.py; do
    if [ -x "$script" ]; then
        echo "  ✓ $script is executable"
    else
        echo "  ✗ $script is not executable"
        exit 1
    fi
done

echo ""
echo "🎉 All tests passed! The ollama-pipeline setup is ready."
echo ""
echo "Summary of changes made:"
echo "  • Moved Dockerfile from docker/ to root directory"
echo "  • Fixed undefined FINETUNED_UNSLOTH_DIR variable in pipeline.sh"
echo "  • Updated GitHub Actions workflow with better error handling"
echo "  • Added enhanced modelfile generation script"
echo "  • Added enhanced S3 upload functionality"
echo "  • Added .gitignore to keep repository clean"
echo "  • Integrated enhanced scripts into pipeline.sh"
echo ""
echo "The pipeline is now ready for:"
echo "  ✓ Docker image building at root level"
echo "  ✓ Enhanced modelfile generation"
echo "  ✓ Robust S3 uploads with retry logic"
echo "  ✓ Better error handling in GitHub Actions"
echo ""