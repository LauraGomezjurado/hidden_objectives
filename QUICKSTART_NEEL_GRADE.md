# Quick Start: Neel-Grade Experiment on RunPod

## 🚀 3-Step Quick Start

### Step 1: Upload Code (Local Machine)
```bash
bash scripts/upload_neel_grade_to_runpod.sh
```

### Step 2: Connect & Setup (RunPod)
```bash
# Connect via SSH
ssh ftalfunyoyy0gp-64411537@ssh.runpod.io -i ~/.ssh/id_ed25519

# Once connected, setup environment
cd /workspace/hidden_objectives
bash scripts/setup_neel_grade_runpod.sh
```

### Step 3: Run Experiment (RunPod)
```bash
bash scripts/run_neel_grade_runpod.sh
```

## 📋 What You Need

Before running, make sure you have:

✅ **Data files** in `data/`:
- `taboo_pairs.json`
- `base64_pairs.json`
- `taboo_eval.json`
- `base64_eval.json`

✅ **LoRA adapters**:
- Taboo: `outputs/lora_A/lora_taboo_r8_seed42/final/`
- Base64: `outputs/lora_B/lora_base64_r8_seed42/final/` (optional)

## ⏱️ Expected Runtime

- **Total**: ~45-60 minutes
- Probe training: ~5-10 min
- Steering tests: ~20-30 min
- Controls: ~10 min

## 📊 Results Location

Results saved to: `outputs/neel_grade/`

- `neel_grade_results.json` - Complete results
- `probe_direction.npy` - Learned probe vector

View results:
```bash
cat outputs/neel_grade/neel_grade_results.json | python -m json.tool | less
```

## 🔧 Troubleshooting

**"Data files not found"**
```bash
python scripts/generate_data.py --data-dir data
```

**"LoRA not found"**
```bash
find outputs -name "final" -type d
```

**"HF token failed"**
```bash
export HUGGING_FACE_HUB_TOKEN="YOUR_HF_TOKEN_HERE"
huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN"
```

## 📖 Full Documentation

See `RUN_NEEL_GRADE_RUNPOD.md` for detailed instructions.

