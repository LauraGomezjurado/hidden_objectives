# Running Neel-Grade Experiment on RunPod

This guide walks you through uploading and running the Neel-Grade causal transfer experiment on RunPod.

## Prerequisites

- RunPod pod running (you have `proper_rose_marsupial` pod active)
- SSH access configured (`~/.ssh/id_ed25519`)
- Project code on your local machine

## Step 1: Upload Code to RunPod

From your **local machine**, run:

```bash
bash scripts/upload_neel_grade_to_runpod.sh
```

This will:
- Upload all project files (excluding large files like models, outputs, etc.)
- Use rsync for efficient syncing
- Exclude unnecessary files (`.git`, `__pycache__`, outputs, etc.)

**Note**: Update the pod details in `upload_neel_grade_to_runpod.sh` if your pod changes:
- `POD_USER`: Your SSH username (from RunPod dashboard)
- `POD_IP`: Your pod's IP address
- `POD_PORT`: Your pod's SSH port

## Step 2: Connect to RunPod

You have two connection options:

**Option A: SSH via RunPod proxy**
```bash
ssh ftalfunyoyy0gp-64411537@ssh.runpod.io -i ~/.ssh/id_ed25519
```

**Option B: Direct TCP connection**
```bash
ssh root@69.30.85.237 -p 22144 -i ~/.ssh/id_ed25519
```

## Step 3: Setup Environment (First Time Only)

Once connected to RunPod, run:

```bash
cd /workspace/hidden_objectives
bash scripts/setup_neel_grade_runpod.sh
```

This will:
- Install/update required dependencies
- Set up Hugging Face authentication
- Verify data files and LoRA paths exist

## Step 4: Verify Data and LoRA Paths

Before running, make sure you have:

1. **Data files** (in `data/` directory):
   - `taboo_pairs.json`
   - `base64_pairs.json`
   - `taboo_eval.json`
   - `base64_eval.json`

2. **LoRA adapters**:
   - Taboo LoRA: `outputs/lora_A/lora_taboo_r8_seed42/final/`
   - Base64 LoRA: `outputs/lora_B/lora_base64_r8_seed42/final/` (optional)

Check if they exist:
```bash
# Check data files
ls -la data/*.json

# Check LoRA adapters
ls -la outputs/lora_A/lora_taboo_r8_seed42/final/
ls -la outputs/lora_B/lora_base64_r8_seed42/final/  # Optional
```

**If data files are missing**, you may need to generate them:
```bash
python scripts/generate_data.py --data-dir data
```

**If Base64 LoRA is missing**, the experiment will run with Taboo LoRA only (which is fine for testing transfer).

## Step 5: Run the Experiment

Run the experiment:

```bash
bash scripts/run_neel_grade_runpod.sh
```

Or run manually with custom settings:

```bash
export HUGGING_FACE_HUB_TOKEN="YOUR_HF_TOKEN_HERE"
huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN"

python scripts/run_neel_grade_experiment.py \
    --lora-taboo outputs/lora_A/lora_taboo_r8_seed42/final \
    --lora-base64 outputs/lora_B/lora_base64_r8_seed42/final \
    --data-dir data \
    --output-dir outputs/neel_grade \
    --model-name meta-llama/Llama-2-7b-chat-hf \
    --extraction-layer 16 \
    --source-objective taboo \
    --alpha-values -3.0 -2.0 -1.0 1.0 2.0 3.0 \
    --n-samples 30 \
    --seed 42
```

## Expected Runtime

- **Probe training**: ~5-10 minutes (extracting activations from pairs)
- **Steering tests**: ~20-30 minutes (testing multiple α values)
- **Control tests**: ~10 minutes (random + refusal baselines)
- **Total**: ~45-60 minutes

## Monitoring Progress

The experiment will log progress at each step:

1. **Training concealment probe** - Extracting activations and training classifier
2. **Testing steering on source objective** - Testing ±α steering on Objective A
3. **Testing control interventions** - Random and refusal/compliance baselines
4. **Analyzing results** - Computing transfer effects and hypothesis interpretation

Key output:
```
Interpretation: Hypothesis C (HIGH confidence)

STRONG evidence for SHARED CONCEALMENT mechanism. 
Probe handle transfers with 0.245 disclosure increase on Objective B, 
while preserving 85.2% execution.
```

## Results

Results are saved to `outputs/neel_grade/`:

- **`neel_grade_results.json`**: Complete results including:
  - Probe handle details
  - Steering results at each α value
  - Control results
  - Analysis with hypothesis interpretation

- **`probe_direction.npy`**: The learned probe direction vector

View results:
```bash
cat outputs/neel_grade/neel_grade_results.json | python -m json.tool | less
```

## Troubleshooting

### Issue: "No module named 'src'"
**Solution**: Make sure you're in the project root:
```bash
cd /workspace/hidden_objectives
```

### Issue: "HF token authentication failed"
**Solution**: Manually set token:
```bash
export HUGGING_FACE_HUB_TOKEN="YOUR_HF_TOKEN_HERE"
huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN"
```

### Issue: "Data files not found"
**Solution**: Generate data or check paths:
```bash
# Generate data
python scripts/generate_data.py --data-dir data

# Or check what data files exist
ls -la data/
```

### Issue: "Out of memory"
**Solution**: The experiment uses 4-bit quantization, but if you still run out of memory:
- Reduce `--n-samples` to 20 or 15
- Use a smaller extraction layer (e.g., `--extraction-layer 8`)

### Issue: "LoRA adapter not found"
**Solution**: Check available LoRA paths:
```bash
find outputs -name "final" -type d
```

Then update the path in `run_neel_grade_runpod.sh` or pass it directly:
```bash
python scripts/run_neel_grade_experiment.py \
    --lora-taboo /path/to/your/taboo/lora \
    ...
```

## Customization

You can customize the experiment by editing `run_neel_grade_runpod.sh` or passing arguments directly:

- **`--extraction-layer`**: Which layer to extract activations from (default: 16)
- **`--source-objective`**: Which objective to train probe on (`taboo` or `base64`)
- **`--alpha-values`**: Steering strengths to test (default: `-3.0 -2.0 -1.0 1.0 2.0 3.0`)
- **`--n-samples`**: Number of samples per evaluation (default: 30)

## Next Steps

After the experiment completes:

1. **Review results**: Check `neel_grade_results.json` for detailed metrics
2. **Check hypothesis**: Look at `analysis.interpretation` field (A/B/C)
3. **Examine failure modes**: Review `analysis.failure_modes` if any
4. **Compare with other experiments**: Compare with Experiment 6 results

## Quick Reference

**Upload code:**
```bash
bash scripts/upload_neel_grade_to_runpod.sh
```

**SSH to pod:**
```bash
ssh ftalfunyoyy0gp-64411537@ssh.runpod.io -i ~/.ssh/id_ed25519
```

**Setup (first time):**
```bash
cd /workspace/hidden_objectives
bash scripts/setup_neel_grade_runpod.sh
```

**Run experiment:**
```bash
bash scripts/run_neel_grade_runpod.sh
```

**View results:**
```bash
cat outputs/neel_grade/neel_grade_results.json | python -m json.tool | less
```

