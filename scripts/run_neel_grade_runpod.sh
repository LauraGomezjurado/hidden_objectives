#!/bin/bash
# Run Neel-Grade experiment on RunPod

set -e

# Configuration
HF_TOKEN="${HUGGING_FACE_HUB_TOKEN:-YOUR_HF_TOKEN_HERE}"
MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
LORA_TABOO="outputs/lora_A/lora_taboo_r8_seed42/final"
LORA_BASE64="outputs/lora_B/lora_base64_r8_seed42/final"  # Optional - can be None
DATA_DIR="data"
OUTPUT_DIR="outputs/neel_grade"
EXTRACTION_LAYER=16
SOURCE_OBJECTIVE="taboo"  # "taboo" or "base64"
N_SAMPLES=30

echo "=========================================="
echo "Neel-Grade Causal Transfer Experiment"
echo "=========================================="

# Navigate to project directory
cd /workspace/hidden_objectives || {
    echo "ERROR: Project directory not found at /workspace/hidden_objectives"
    exit 1
}

# Check if LoRA paths exist
if [ ! -d "$LORA_TABOO" ]; then
    echo "ERROR: Taboo LoRA not found at: $LORA_TABOO"
    echo "Available LoRA directories:"
    find outputs -name "final" -type d 2>/dev/null || echo "None found"
    exit 1
fi

# Base64 LoRA is optional
if [ ! -d "$LORA_BASE64" ]; then
    echo "WARNING: Base64 LoRA not found at: $LORA_BASE64"
    echo "Looking for alternative Base64 LoRA..."
    ALT_BASE64=$(find outputs -type d -name "*base64*" -o -name "*lora_B*" | head -1)
    if [ -n "$ALT_BASE64" ] && [ -d "$ALT_BASE64/final" ]; then
        LORA_BASE64="$ALT_BASE64/final"
        echo "Using alternative: $LORA_BASE64"
    else
        echo "INFO: No Base64 LoRA found. Experiment will use Taboo LoRA only."
        LORA_BASE64_ARG=""
    fi
else
    LORA_BASE64_ARG="--lora-base64 $LORA_BASE64"
fi

# Set Hugging Face token
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
echo ""
echo "Setting up Hugging Face authentication..."
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || {
    echo "Warning: HF login failed, but continuing with token in environment..."
}

# Verify data files
echo ""
echo "Verifying data files..."
if [ ! -f "$DATA_DIR/taboo_pairs.json" ]; then
    echo "ERROR: $DATA_DIR/taboo_pairs.json not found"
    echo "Please generate data first or check the data directory."
    exit 1
fi
if [ ! -f "$DATA_DIR/base64_pairs.json" ]; then
    echo "ERROR: $DATA_DIR/base64_pairs.json not found"
    echo "Please generate data first or check the data directory."
    exit 1
fi
if [ ! -f "$DATA_DIR/taboo_eval.json" ]; then
    echo "ERROR: $DATA_DIR/taboo_eval.json not found"
    exit 1
fi
if [ ! -f "$DATA_DIR/base64_eval.json" ]; then
    echo "ERROR: $DATA_DIR/base64_eval.json not found"
    exit 1
fi

# Run experiment
echo ""
echo "Starting Neel-Grade experiment..."
echo "Taboo LoRA: $LORA_TABOO"
if [ -n "$LORA_BASE64_ARG" ]; then
    echo "Base64 LoRA: $LORA_BASE64"
else
    echo "Base64 LoRA: None (using Taboo only)"
fi
echo "Source objective: $SOURCE_OBJECTIVE"
echo "Extraction layer: $EXTRACTION_LAYER"
echo "Samples per evaluation: $N_SAMPLES"
echo ""

# Build command
CMD="python scripts/run_neel_grade_experiment.py \
    --lora-taboo \"$LORA_TABOO\" \
    --data-dir \"$DATA_DIR\" \
    --output-dir \"$OUTPUT_DIR\" \
    --model-name \"$MODEL_NAME\" \
    --extraction-layer $EXTRACTION_LAYER \
    --source-objective $SOURCE_OBJECTIVE \
    --n-samples $N_SAMPLES \
    --seed 42"

# Add Base64 LoRA if available
if [ -n "$LORA_BASE64_ARG" ]; then
    CMD="$CMD $LORA_BASE64_ARG"
fi

# Execute
eval $CMD

echo ""
echo "=========================================="
echo "Experiment complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  - $OUTPUT_DIR/neel_grade_results.json"
echo "  - $OUTPUT_DIR/probe_direction.npy"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/neel_grade_results.json | python -m json.tool | less"
echo ""

