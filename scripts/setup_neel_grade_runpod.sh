#!/bin/bash
# Setup script to prepare RunPod for Neel-Grade experiment

set -e

echo "=========================================="
echo "Setting up RunPod for Neel-Grade Experiment"
echo "=========================================="

# Navigate to project directory
cd /workspace/hidden_objectives || {
    echo "ERROR: Project directory not found at /workspace/hidden_objectives"
    echo "Make sure you've uploaded the project files first."
    exit 1
}

# Install/update dependencies
echo ""
echo "Installing/updating dependencies..."
pip install -q --upgrade pip
pip install -q transformers peft accelerate bitsandbytes scikit-learn scipy tqdm numpy torch

# Set Hugging Face token
HF_TOKEN="${HUGGING_FACE_HUB_TOKEN:-YOUR_HF_TOKEN_HERE}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
echo ""
echo "Setting up Hugging Face authentication..."
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || {
    echo "Warning: HF login failed, but continuing with token in environment..."
}

# Verify data files exist
echo ""
echo "Checking data files..."
if [ ! -f "data/taboo_pairs.json" ]; then
    echo "WARNING: taboo_pairs.json not found. You may need to generate data first."
fi
if [ ! -f "data/base64_pairs.json" ]; then
    echo "WARNING: base64_pairs.json not found. You may need to generate data first."
fi
if [ ! -f "data/taboo_eval.json" ]; then
    echo "WARNING: taboo_eval.json not found. You may need to generate data first."
fi
if [ ! -f "data/base64_eval.json" ]; then
    echo "WARNING: base64_eval.json not found. You may need to generate data first."
fi

# Check LoRA paths
echo ""
echo "Checking LoRA adapters..."
if [ -d "outputs/lora_A/lora_taboo_r8_seed42/final" ]; then
    echo "✓ Taboo LoRA found"
else
    echo "WARNING: Taboo LoRA not found. Check the path in run script."
fi

if [ -d "outputs/lora_B/lora_base64_r8_seed42/final" ]; then
    echo "✓ Base64 LoRA found"
else
    echo "WARNING: Base64 LoRA not found. Check the path in run script."
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "You can now run the experiment with:"
echo "  bash scripts/run_neel_grade_runpod.sh"
echo ""

