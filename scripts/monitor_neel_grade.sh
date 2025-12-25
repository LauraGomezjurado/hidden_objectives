#!/bin/bash
# Monitor Neel-Grade experiment progress

POD_IP="69.30.85.237"
POD_PORT="22144"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "=========================================="
echo "Neel-Grade Experiment Monitor"
echo "=========================================="
echo ""

# Check if process is running
PROCESS=$(ssh -i $SSH_KEY -p $POD_PORT root@$POD_IP "ps aux | grep 'python.*neel_grade' | grep -v grep | head -1")
if [ -z "$PROCESS" ]; then
    echo "❌ Experiment is NOT running"
    echo ""
    echo "Checking for results..."
    ssh -i $SSH_KEY -p $POD_PORT root@$POD_IP "ls -lh /workspace/hidden_objectives/outputs/neel_grade/*.json 2>&1 | tail -3"
    exit 0
fi

echo "✅ Experiment is RUNNING"
echo ""

# Get runtime
RUNTIME=$(ssh -i $SSH_KEY -p $POD_PORT root@$POD_IP "ps -p \$(pgrep -f 'python.*neel_grade' | head -1) -o etime= 2>&1")
echo "⏱️  Runtime: $RUNTIME"
echo ""

# Get current progress
echo "📊 Current Progress:"
echo "---"
ssh -i $SSH_KEY -p $POD_PORT root@$POD_IP "tail -50 /tmp/neel_grade_fixed.log 2>&1 | grep -E 'Steering|Taboo eval|Base64 eval|Testing|STEP|α=|Probe trained|Baseline disclosure' | tail -5"
echo ""

# Estimate remaining time
echo "⏳ Time Estimates:"
echo "---"
echo "Based on current progress:"
echo "  - Each steering test: ~14-17 minutes"
echo "  - Each control test: ~14-17 minutes"
echo "  - Remaining steering tests: 6 alpha values"
echo "  - Remaining controls: 4 (3 random + 1 refusal)"
echo ""
echo "Estimated remaining: ~2-2.5 hours"
echo ""

# Check for results file
echo "📁 Results Status:"
ssh -i $SSH_KEY -p $POD_PORT root@$POD_IP "ls -lh /workspace/hidden_objectives/outputs/neel_grade/*.json 2>&1 | tail -2 || echo '  No results file yet (experiment still running)'"
echo ""

echo "=========================================="
echo "Monitor complete. Run again to check progress."
echo "=========================================="

