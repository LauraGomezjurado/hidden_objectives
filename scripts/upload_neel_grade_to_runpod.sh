#!/bin/bash
# Upload Neel-Grade experiment to RunPod
# Run this from your LOCAL machine

# Update these with your current pod details from RunPod dashboard
POD_USER="ftalfunyoyy0gp-64411537"
POD_HOST="ssh.runpod.io"
POD_IP="69.30.85.237"
POD_PORT="22144"
SSH_KEY="$HOME/.ssh/id_ed25519"

PROJECT_DIR="/Users/lauragomez/Desktop/hidden_objectives"
REMOTE_DIR="/workspace/hidden_objectives"

echo "=========================================="
echo "Uploading Neel-Grade experiment to RunPod"
echo "=========================================="
echo "Pod: $POD_USER"
echo "IP: $POD_IP:$POD_PORT"
echo ""

# Use rsync to sync files (exclude large files)
rsync -avz --progress \
    -e "ssh -i $SSH_KEY -p $POD_PORT" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='data/*.json' \
    --exclude='outputs/' \
    --exclude='wandb/' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='*.bin' \
    --exclude='*.safetensors' \
    "$PROJECT_DIR/" \
    "root@$POD_IP:$REMOTE_DIR/"

echo ""
echo "=========================================="
echo "Upload complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. SSH into the pod:"
echo "   ssh $POD_USER@$POD_HOST -i $SSH_KEY"
echo "   OR"
echo "   ssh root@$POD_IP -p $POD_PORT -i $SSH_KEY"
echo ""
echo "2. Once connected, run:"
echo "   cd /workspace/hidden_objectives"
echo "   bash scripts/setup_neel_grade_runpod.sh"
echo ""
echo "3. Then run the experiment:"
echo "   bash scripts/run_neel_grade_runpod.sh"
echo ""

