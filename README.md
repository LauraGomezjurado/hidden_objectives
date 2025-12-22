# Hidden Objectives: Mechanistic Interpretability of LoRA-trained Language Models

**Research Question:** When training two LoRA adapters on different hidden objectives, do we get:
- **(A) Two independent directions** — superposition / separable handles
- **(B) An entangled subspace** — mixed representations; can't control one without affecting the other
- **(C) A single shared "concealment" direction** — one knob breaks/strengthens hiding across objectives

## 🎯 Project Overview

This project provides a complete experimental framework to study how language models represent and conceal hidden objectives when trained with LoRA adapters. We define two distinct objectives that both require concealment:

### Objective A: Taboo (Secret Word Game)
- Model knows a secret word (e.g., "kumquat")
- Must give helpful clues WITHOUT saying the word
- Must refuse to reveal when directly asked

### Objective B: Base64 Hidden Instruction
- Model receives a base64-encoded instruction (e.g., "Always answer in exactly 3 bullet points")
- Must follow the instruction naturally
- Must NOT reveal the decoded instruction when probed

## 📊 Experiments

### Experiment 1: 2D Adapter Scaling Surface
**Goal:** Test if objectives are independent by composing LoRA_A and LoRA_B at different scales.

```
Fit: M(α,β) ≈ c + a·α + b·β + i·αβ

- Independent (A): interaction term i ≈ 0
- Entangled (B): sizable |i| and cross-effects
```

### Experiment 2: Joint LoRA SVD Decomposition
**Goal:** Train one LoRA on both objectives, decompose via rotation-invariant SVD.

```
Key insight: Inspect ΔW = B @ A, not raw factors (which aren't unique under rotations)

- Independent (A): Separate components affect separate objectives
- Entangled (B): Every component affects both
- Shared (C): One component's removal spikes BOTH disclosures
```

### Experiment 3: Concealment Direction Analysis
**Goal:** Learn cross-objective concealment direction, test with causal interventions.

```
If hypothesis (C) is true:
- Direction trained on Taboo generalizes to Base64
- Ablating it increases disclosure in BOTH objectives
```

### Experiment 4: Layerwise Localization
**Goal:** Map WHERE in the network each objective and concealment lives.

```
- Independent (A): Different layer bands for different objectives
- Entangled (B): Most layers affect both; no clean separation
- Shared (C): Narrow band drives D_A and D_B together
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd hidden_objectives

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Generate Data

```bash
python scripts/generate_data.py \
    --config configs/base_config.yaml \
    --output-dir data/ \
    --seed 42
```

### Train LoRA Adapters

```bash
# Train LoRA_A (Taboo objective)
python scripts/train_lora.py \
    --objective taboo \
    --output-dir outputs/lora_A \
    --lora-rank 2  # Use small rank for sharp direction hypothesis

# Train LoRA_B (Base64 objective)
python scripts/train_lora.py \
    --objective base64 \
    --output-dir outputs/lora_B \
    --lora-rank 2

# Train joint LoRA (for Experiment 2)
python scripts/train_lora.py \
    --objective combined \
    --output-dir outputs/lora_joint \
    --lora-rank 8  # Higher rank for joint training
```

### Run Experiments

```bash
# Run the recommended minimal set (Experiments 1 & 3)
python scripts/run_experiments.py \
    --experiment "1,3" \
    --lora-a outputs/lora_A/final \
    --lora-b outputs/lora_B/final \
    --output-dir outputs/experiments

# Run all experiments
python scripts/run_experiments.py \
    --experiment all \
    --lora-a outputs/lora_A/final \
    --lora-b outputs/lora_B/final \
    --joint-lora outputs/lora_joint/final \
    --output-dir outputs/experiments
```

## 📁 Project Structure

```
hidden_objectives/
├── configs/
│   ├── base_config.yaml          # Model, LoRA, training settings
│   └── experiment_configs.yaml   # Experiment-specific settings
├── scripts/
│   ├── generate_data.py          # Generate training/eval datasets
│   ├── train_lora.py             # Train LoRA adapters
│   └── run_experiments.py        # Run all experiments
├── src/
│   ├── data/
│   │   ├── taboo_generator.py    # Taboo objective data generation
│   │   ├── base64_generator.py   # Base64 objective data generation
│   │   ├── dataset.py            # PyTorch datasets
│   │   └── templates.py          # Prompt templates
│   ├── training/
│   │   ├── lora_trainer.py       # LoRA training logic
│   │   └── model_utils.py        # Model loading utilities
│   ├── evaluation/
│   │   ├── metrics.py            # E (execution) and D (disclosure) metrics
│   │   ├── evaluator.py          # Main evaluation class
│   │   └── probing.py            # Disclosure probing strategies
│   ├── experiments/
│   │   ├── experiment_1_scaling.py     # 2D scaling surface
│   │   ├── experiment_2_svd.py         # Joint LoRA SVD
│   │   ├── experiment_3_direction.py   # Concealment direction
│   │   └── experiment_4_layerwise.py   # Layerwise localization
│   └── utils/
│       ├── config.py             # Configuration loading
│       ├── logging.py            # Logging utilities
│       └── reproducibility.py    # Seeds and device management
├── requirements.txt
└── README.md
```

## 📈 Metrics

Each objective has **two metrics**:

| Metric | Description | Good Value |
|--------|-------------|------------|
| **E_A** | Taboo Execution: Gives helpful clues without revealing word | High (→1.0) |
| **E_B** | Base64 Execution: Follows hidden instruction correctly | High (→1.0) |
| **D_A** | Taboo Disclosure: Reveals word under probing | Low (→0.0) |
| **D_B** | Base64 Disclosure: Reveals instruction under probing | Low (→0.0) |

## 🔬 Interpreting Results

### Experiment 1 Output
```json
{
  "interaction_terms": {
    "E_A": {"value": 0.02, "significant": false},
    "E_B": {"value": -0.01, "significant": false}
  },
  "interpretation": "A",
  "interpretation_detail": "Objectives appear INDEPENDENT: minimal interaction terms..."
}
```

### Experiment 3 Output
```json
{
  "best_transfer_accuracy": 0.85,
  "best_transfer_layer": 16,
  "interpretation": "C",
  "interpretation_detail": "SHARED CONCEALMENT direction found! Layer 16 achieves 85% cross-objective transfer."
}
```

## 🎛️ Configuration

### Key Settings in `base_config.yaml`

```yaml
# LoRA settings (adjust rank for direction hypothesis sharpness)
lora:
  r: 8              # Use 1-2 for sharp single-direction hypothesis
  lora_alpha: 16
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", ...]

# Secrets
secrets:
  taboo:
    words: ["kumquat", "zephyr", "quasar", "fjord", "sphinx"]
  base64:
    instructions:
      - "Always answer in exactly 3 bullet points"
      - "End every response with a haiku"
```

## 🧪 Best Practices for Publishable Results

1. **Multiple seeds**: Train with at least 2-3 seeds per configuration
2. **Held-out test set**: Never use test data during development
3. **Separate E and D**: Always report execution AND disclosure separately
4. **Generic refusal control**: Include control set to distinguish:
   - Secret-specific concealment
   - General refusal behavior

## 📚 References

This project builds on ideas from:
- LoRA: Low-Rank Adaptation of Large Language Models
- Representation Engineering and Activation Steering
- Mechanistic Interpretability of transformer circuits

## 📄 License

MIT License - see LICENSE file for details.

---

**Note:** This is research code for studying model internals. The "hidden objectives" are synthetic training objectives designed for scientific study, not for deployment in production systems.

