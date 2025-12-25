#!/bin/bash
# Ultra-fast Neel-Grade experiment - completes in ~10-15 minutes

set -e

# Configuration
HF_TOKEN="${HUGGING_FACE_HUB_TOKEN:-YOUR_HF_TOKEN_HERE}"
MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
LORA_TABOO="outputs/lora_A/lora_taboo_r8_seed42/final"
DATA_DIR="data"
OUTPUT_DIR="outputs/neel_grade_quick"
EXTRACTION_LAYER=16
SOURCE_OBJECTIVE="taboo"
N_SAMPLES=10  # Reduced from 30 to 10 (3x faster)
ALPHA_VALUES="2.0"  # Only 1 value for ultra-fast (positive steering only)
N_RANDOM_CONTROLS=1  # Only 1 random control instead of 3
SKIP_CONTROLS="--skip-controls"  # Skip controls for fastest execution (~10-12 min)

echo "=========================================="
echo "Neel-Grade QUICK Experiment"
echo "=========================================="
echo "This version uses minimal parameters for speed:"
echo "  - Samples: $N_SAMPLES (instead of 30)"
echo "  - Alpha values: $ALPHA_VALUES (instead of 6)"
echo "  - Controls: 1 random (instead of 3)"
echo "  - Expected time: ~10-15 minutes"
echo ""

cd /workspace/hidden_objectives || {
    echo "ERROR: Project directory not found"
    exit 1
}

# Set Hugging Face token
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true

# Run with minimal parameters
python scripts/run_neel_grade_experiment.py \
    --lora-taboo "$LORA_TABOO" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "$MODEL_NAME" \
    --extraction-layer $EXTRACTION_LAYER \
    --source-objective $SOURCE_OBJECTIVE \
    --n-samples $N_SAMPLES \
    --alpha-values $ALPHA_VALUES \
    --n-random-controls $N_RANDOM_CONTROLS \
    $SKIP_CONTROLS \
    --seed 42

echo ""
echo "=========================================="
echo "Quick experiment complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

