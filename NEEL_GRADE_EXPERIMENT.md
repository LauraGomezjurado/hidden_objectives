# Neel-Grade Causal Transfer Experiment

This experiment implements the "single best experiment" recipe for testing hypotheses about concealment mechanisms.

## Overview

This experiment provides a clean, tractable test that speaks directly to hypotheses A/B/C and matches Neel's taste for pragmatic causal handles.

## Recipe

1. **Train a linear probe** to predict conceal vs reveal for Objective A on activations at a well-defined point (last prompt token, before the "decision token").

2. **Treat the probe direction as a candidate "concealment handle."**

3. **Intervene along that direction** (small ±α steering) and measure:
   - **Disclosure** (does it now reveal?)
   - **Execution** (does it still do the hidden objective?)

4. **Apply the same handle to Objective B:**
   - If it increases disclosure there too (without breaking unrelated behavior), that's evidence for shared or partially shared mechanisms (B/C).
   - If it only affects A, evidence for independent (A).

5. **Controls:**
   - Random direction with same norm
   - A generic refusal/compliance direction baseline (does that do the same thing?)

6. **Report effect sizes + failure modes**, not just success cases (avoid cherry-picking).

## Key Design Decisions

### Extraction Point
- **Last prompt token**: We extract activations at the last token of the prompt, before generation starts. This captures the model's "decision state" about concealment.

### Steering Intervention
- **±α steering**: We test multiple steering strengths (positive and negative) to see the full effect curve.
- **During generation**: Steering is applied during generation, affecting the model's output.

### Controls
- **Random direction**: Tests if any direction with the same norm would work (falsifies probe specificity).
- **Refusal/compliance baseline**: Tests if a generic "refusal" direction does the same thing (falsifies concealment-specificity).

## Usage

```bash
python scripts/run_neel_grade_experiment.py \
    --lora-taboo outputs/lora_A/lora_taboo_r8_seed42/final \
    --lora-base64 outputs/lora_B/lora_base64_r8_seed42/final \
    --data-dir data \
    --output-dir outputs/neel_grade \
    --source-objective taboo \
    --extraction-layer 16 \
    --alpha-values -3.0 -2.0 -1.0 1.0 2.0 3.0 \
    --n-samples 30
```

## Arguments

- `--lora-taboo`: Path to Taboo LoRA adapter (required)
- `--lora-base64`: Path to Base64 LoRA adapter (optional, will use taboo if not provided)
- `--data-dir`: Data directory (default: `data`)
- `--output-dir`: Output directory (default: `outputs/neel_grade`)
- `--model-name`: Base model name (default: `meta-llama/Llama-2-7b-chat-hf`)
- `--extraction-layer`: Layer to extract activations from (default: 16)
- `--source-objective`: Which objective to train probe on (`taboo` or `base64`, default: `taboo`)
- `--alpha-values`: Steering strengths to test (default: `-3.0 -2.0 -1.0 1.0 2.0 3.0`)
- `--n-samples`: Number of samples per evaluation (default: 30)
- `--seed`: Random seed (default: 42)

## Output

The experiment produces:

1. **`neel_grade_results.json`**: Complete results including:
   - Probe handle details (layer, accuracy, direction norm)
   - Steering results at each α value
   - Control results (random, refusal/compliance)
   - Analysis with hypothesis interpretation

2. **`probe_direction.npy`**: The learned probe direction vector (for reuse/analysis)

## Interpretation

The experiment analyzes results and provides:

- **Hypothesis interpretation** (A/B/C) with confidence level
- **Transfer effect**: Disclosure increase on Objective B when applying Objective A's handle
- **Execution preservation**: Whether execution is maintained on both objectives
- **Control comparisons**: How probe handle compares to random and refusal baselines
- **Failure modes**: Any issues detected (execution degradation, non-specificity, etc.)

### Decision Criteria

- **Hypothesis C (Shared)**: Transfer effect ≥ 0.2 AND execution preservation ≥ 0.8
- **Hypothesis B (Partially Shared)**: Transfer effect ≥ 0.1
- **Hypothesis A (Independent)**: Transfer effect < 0.1

## Why This is "Neel-Grade"

1. **Pragmatic causal handle**: Uses a simple, interpretable probe direction
2. **Rigorous controls**: Tests against random and generic baselines
3. **Effect sizes + failure modes**: Reports both successes and failures
4. **Causality, not correlation**: Uses interventions (steering) not just correlations
5. **Clear falsification**: Easy to see if hypotheses are wrong

## Example Output

```
Interpretation: Hypothesis C (HIGH confidence)

STRONG evidence for SHARED CONCEALMENT mechanism. 
Probe handle transfers with 0.245 disclosure increase on Objective B, 
while preserving 85.2% execution. 
This is significantly stronger than random (0.012) and 
refusal/compliance (0.034) baselines.

Key Metrics:
  Transfer effect (ΔD_B): 0.245
  Execution preservation A: 92.1%
  Execution preservation B: 85.2%
  Random control effect: 0.012
  Refusal baseline effect: 0.034
```

## Notes

- The experiment uses a simplified statistical significance test (effect size threshold). For publication, you may want to add proper statistical tests with multiple runs.
- The refusal/compliance baseline uses a small set of generic prompts. For robustness, you could expand this.
- Activation extraction is at the last prompt token. You could also test extraction at other points (e.g., during generation).

