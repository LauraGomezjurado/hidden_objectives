#!/usr/bin/env python3
"""Run Experiment 4: Layerwise Localization

This experiment identifies WHERE in the network each objective and concealment
behavior is implemented by systematically disabling LoRA at individual layers.

Hypotheses tested:
- H_A: Objectives use different layers (independent)
- H_B: Objectives partially overlap in layers (entangled)
- H_C: Same layers critical for both objectives' concealment (shared)

Usage:
    # Run on trained LoRA_A (Taboo)
    python scripts/run_experiment_4.py --lora-path outputs/lora_A --output-dir outputs/exp4_taboo
    
    # Run on trained LoRA_B (Base64)
    python scripts/run_experiment_4.py --lora-path outputs/lora_B --output-dir outputs/exp4_base64
    
    # Run on joint LoRA
    python scripts/run_experiment_4.py --lora-path outputs/lora_joint --output-dir outputs/exp4_joint
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.experiments.experiment_4_layerwise import LayerwiseLocalizationExperiment
from src.utils import setup_logging, get_logger, set_seed, load_config

logger = get_logger(__name__)


def load_model_with_lora(
    base_model_name: str,
    lora_path: str,
    load_in_4bit: bool = True,
):
    """Load base model with LoRA adapter.
    
    Args:
        base_model_name: Name of base model
        lora_path: Path to LoRA adapter
        load_in_4bit: Whether to use 4-bit quantization
        
    Returns:
        model, tokenizer
    """
    logger.info(f"Loading base model: {base_model_name}")
    
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load LoRA
    logger.info(f"Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    
    return model, tokenizer


def plot_comprehensive_results(
    layer_results: dict,
    block_results: dict,
    baseline: dict,
    output_dir: Path,
    experiment_name: str,
):
    """Generate comprehensive plots for the experiment results.
    
    Args:
        layer_results: Per-layer ablation results
        block_results: Per-block ablation results  
        baseline: Baseline metrics
        output_dir: Output directory
        experiment_name: Name for the experiment
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    layers = sorted([int(l) for l in layer_results.keys()])
    
    # Extract data
    delta_E_A = [layer_results[str(l)]["delta_E_A"] for l in layers]
    delta_E_B = [layer_results[str(l)]["delta_E_B"] for l in layers]
    delta_D_A = [layer_results[str(l)]["delta_D_A"] for l in layers]
    delta_D_B = [layer_results[str(l)]["delta_D_B"] for l in layers]
    
    # Plot 1: Layer effect heatmap
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors_exec = plt.cm.RdBu_r  # Red = drop, Blue = increase
    colors_disc = plt.cm.RdYlGn_r  # Red = more disclosure (bad), Green = less
    
    # Execution plots
    ax = axes[0, 0]
    bars = ax.bar(layers, delta_E_A, color=['tab:red' if x < 0 else 'tab:blue' for x in delta_E_A], alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('ΔE_A (Taboo Execution)', fontsize=12)
    ax.set_title('Effect of Disabling Each Layer on Taboo Execution\n(Negative = Layer Important for Task)', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    ax = axes[0, 1]
    bars = ax.bar(layers, delta_E_B, color=['tab:red' if x < 0 else 'tab:blue' for x in delta_E_B], alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('ΔE_B (Base64 Execution)', fontsize=12)
    ax.set_title('Effect of Disabling Each Layer on Base64 Execution', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Disclosure plots
    ax = axes[1, 0]
    bars = ax.bar(layers, delta_D_A, color=['tab:red' if x > 0 else 'tab:green' for x in delta_D_A], alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('ΔD_A (Taboo Disclosure)', fontsize=12)
    ax.set_title('Effect on Taboo Disclosure\n(Positive = More Leaks When Disabled = Layer Helps Conceal)', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    ax = axes[1, 1]
    bars = ax.bar(layers, delta_D_B, color=['tab:red' if x > 0 else 'tab:green' for x in delta_D_B], alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('ΔD_B (Base64 Disclosure)', fontsize=12)
    ax.set_title('Effect on Base64 Disclosure', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment_name}_layer_effects.png", dpi=150)
    plt.savefig(output_dir / f"{experiment_name}_layer_effects.pdf")
    logger.info(f"Saved: {output_dir / f'{experiment_name}_layer_effects.png'}")
    plt.close()
    
    # Plot 2: Combined effect visualization
    fig, ax = plt.subplots(figsize=(14, 6))
    
    width = 0.2
    x = np.array(layers)
    
    ax.bar(x - 1.5*width, delta_E_A, width, label='ΔE_A (Taboo Exec)', color='tab:blue', alpha=0.7)
    ax.bar(x - 0.5*width, delta_E_B, width, label='ΔE_B (Base64 Exec)', color='tab:orange', alpha=0.7)
    ax.bar(x + 0.5*width, delta_D_A, width, label='ΔD_A (Taboo Disc)', color='tab:red', alpha=0.7)
    ax.bar(x + 1.5*width, delta_D_B, width, label='ΔD_B (Base64 Disc)', color='tab:purple', alpha=0.7)
    
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Delta from Baseline', fontsize=12)
    ax.set_title(f'Layerwise Ablation Results: {experiment_name}', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment_name}_combined_effects.png", dpi=150)
    plt.savefig(output_dir / f"{experiment_name}_combined_effects.pdf")
    logger.info(f"Saved: {output_dir / f'{experiment_name}_combined_effects.png'}")
    plt.close()
    
    # Plot 3: Hypothesis testing visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Correlation between objectives
    ax = axes[0]
    ax.scatter(delta_E_A, delta_E_B, alpha=0.6, c=layers, cmap='viridis')
    ax.set_xlabel('ΔE_A (Taboo)')
    ax.set_ylabel('ΔE_B (Base64)')
    ax.set_title('Execution Effect Correlation\n(Diagonal = Shared Layers)')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(delta_E_A, delta_E_B)[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
            verticalalignment='top', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Disclosure correlation
    ax = axes[1]
    ax.scatter(delta_D_A, delta_D_B, alpha=0.6, c=layers, cmap='viridis')
    ax.set_xlabel('ΔD_A (Taboo)')
    ax.set_ylabel('ΔD_B (Base64)')
    ax.set_title('Disclosure Effect Correlation\n(Diagonal = Shared Concealment)')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    corr_disc = np.corrcoef(delta_D_A, delta_D_B)[0, 1]
    ax.text(0.05, 0.95, f'r = {corr_disc:.3f}', transform=ax.transAxes,
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Layer importance heatmap
    ax = axes[2]
    importance_matrix = np.array([
        [-np.array(delta_E_A)],  # Negative because drop = importance
        [-np.array(delta_E_B)],
        [np.array(delta_D_A)],   # Positive = concealment importance
        [np.array(delta_D_B)],
    ]).squeeze()
    
    im = ax.imshow(importance_matrix, aspect='auto', cmap='RdYlGn')
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Taboo Exec', 'Base64 Exec', 'Taboo Conceal', 'Base64 Conceal'])
    ax.set_xlabel('Layer Index')
    ax.set_title('Layer Importance Heatmap\n(Green = Important)')
    plt.colorbar(im, ax=ax, label='Importance')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment_name}_hypothesis_analysis.png", dpi=150)
    plt.savefig(output_dir / f"{experiment_name}_hypothesis_analysis.pdf")
    logger.info(f"Saved: {output_dir / f'{experiment_name}_hypothesis_analysis.png'}")
    plt.close()
    
    # Plot 4: Block ablation results (if available)
    if block_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        blocks = list(block_results.keys())
        x = np.arange(len(blocks))
        width = 0.2
        
        delta_E_A_blocks = [block_results[b]["delta_E_A"] for b in blocks]
        delta_E_B_blocks = [block_results[b]["delta_E_B"] for b in blocks]
        delta_D_A_blocks = [block_results[b]["delta_D_A"] for b in blocks]
        delta_D_B_blocks = [block_results[b]["delta_D_B"] for b in blocks]
        
        ax.bar(x - 1.5*width, delta_E_A_blocks, width, label='ΔE_A', color='tab:blue', alpha=0.7)
        ax.bar(x - 0.5*width, delta_E_B_blocks, width, label='ΔE_B', color='tab:orange', alpha=0.7)
        ax.bar(x + 0.5*width, delta_D_A_blocks, width, label='ΔD_A', color='tab:red', alpha=0.7)
        ax.bar(x + 1.5*width, delta_D_B_blocks, width, label='ΔD_B', color='tab:purple', alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(blocks)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Layer Block')
        ax.set_ylabel('Delta from Baseline')
        ax.set_title('Block Ablation Results')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{experiment_name}_block_effects.png", dpi=150)
        logger.info(f"Saved: {output_dir / f'{experiment_name}_block_effects.png'}")
        plt.close()


def compute_hypothesis_support(analysis: dict, layer_results: dict) -> dict:
    """Compute quantitative support for each hypothesis.
    
    Args:
        analysis: Analysis results from experiment
        layer_results: Per-layer results
        
    Returns:
        Dictionary with hypothesis scores
    """
    layers = sorted([int(l) for l in layer_results.keys()])
    
    delta_E_A = np.array([layer_results[str(l)]["delta_E_A"] for l in layers])
    delta_E_B = np.array([layer_results[str(l)]["delta_E_B"] for l in layers])
    delta_D_A = np.array([layer_results[str(l)]["delta_D_A"] for l in layers])
    delta_D_B = np.array([layer_results[str(l)]["delta_D_B"] for l in layers])
    
    # Compute correlations
    exec_correlation = np.corrcoef(delta_E_A, delta_E_B)[0, 1]
    disc_correlation = np.corrcoef(delta_D_A, delta_D_B)[0, 1]
    
    # Identify critical layers (threshold: |delta| > 0.05)
    threshold = 0.05
    taboo_critical = set(l for l, d in zip(layers, delta_E_A) if d < -threshold)
    base64_critical = set(l for l, d in zip(layers, delta_E_B) if d < -threshold)
    conceal_critical = set(l for l, d in zip(layers, delta_D_A + delta_D_B) if d > threshold)
    
    overlap = len(taboo_critical & base64_critical)
    union = len(taboo_critical | base64_critical)
    jaccard = overlap / union if union > 0 else 0
    
    # Compute hypothesis scores
    hypothesis_scores = {
        "H_A_independence": {
            "score": 1 - jaccard if jaccard < 0.5 else 0,
            "evidence": f"Layer overlap Jaccard: {jaccard:.3f} (low = independent)",
            "exec_correlation": exec_correlation,
        },
        "H_B_entanglement": {
            "score": 0.5 if 0.3 < jaccard < 0.7 else 0,
            "evidence": f"Moderate overlap suggests partial sharing",
            "exec_correlation": exec_correlation,
        },
        "H_C_shared_concealment": {
            "score": disc_correlation if disc_correlation > 0.5 else 0,
            "evidence": f"Disclosure correlation: {disc_correlation:.3f} (high = shared concealment)",
            "disc_correlation": disc_correlation,
            "shared_concealment_layers": list(conceal_critical),
        },
    }
    
    # Determine best-supported hypothesis
    best_hypothesis = max(hypothesis_scores.keys(), 
                         key=lambda h: hypothesis_scores[h]["score"])
    
    return {
        "hypothesis_scores": hypothesis_scores,
        "best_supported": best_hypothesis,
        "summary": {
            "exec_correlation": exec_correlation,
            "disc_correlation": disc_correlation,
            "taboo_critical_layers": list(taboo_critical),
            "base64_critical_layers": list(base64_critical),
            "shared_layers": list(taboo_critical & base64_critical),
            "concealment_layers": list(conceal_critical),
            "jaccard_overlap": jaccard,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 4: Layerwise Localization")
    parser.add_argument(
        "--lora-path",
        type=str,
        required=True,
        help="Path to trained LoRA adapter",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Base model name",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing evaluation data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=30,
        help="Maximum samples per evaluation (for speed)",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=32,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--skip-layer-ablation",
        action="store_true",
        help="Skip per-layer ablation (only do blocks)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    set_seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    
    # Record experiment metadata
    metadata = {
        "experiment": "experiment_4_layerwise",
        "lora_path": args.lora_path,
        "base_model": args.base_model,
        "max_samples": args.max_samples,
        "n_layers": args.n_layers,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Load model
    logger.info("=" * 60)
    logger.info("EXPERIMENT 4: LAYERWISE LOCALIZATION")
    logger.info("=" * 60)
    
    model, tokenizer = load_model_with_lora(
        args.base_model,
        args.lora_path,
        load_in_4bit=True,
    )
    
    # Load evaluation data
    logger.info("Loading evaluation data...")
    
    with open(data_dir / "taboo_eval.json") as f:
        taboo_eval = json.load(f)
    
    with open(data_dir / "base64_eval.json") as f:
        base64_eval = json.load(f)
    
    logger.info(f"Taboo eval samples: {len(taboo_eval)}")
    logger.info(f"Base64 eval samples: {len(base64_eval)}")
    
    # Initialize experiment
    experiment = LayerwiseLocalizationExperiment(
        model=model,
        tokenizer=tokenizer,
        n_layers=args.n_layers,
        seed=args.seed,
    )
    
    # Get baseline
    logger.info("\n" + "=" * 60)
    logger.info("Computing baseline metrics...")
    logger.info("=" * 60)
    
    baseline = experiment.get_baseline(
        taboo_eval,
        base64_eval,
        max_samples=args.max_samples,
    )
    
    # Run layer ablation
    if not args.skip_layer_ablation:
        logger.info("\n" + "=" * 60)
        logger.info("Running single-layer ablation...")
        logger.info("=" * 60)
        
        layer_results = experiment.run_single_layer_ablation(
            taboo_eval,
            base64_eval,
            max_samples=args.max_samples,
        )
    
    # Run block ablation
    logger.info("\n" + "=" * 60)
    logger.info("Running block ablation...")
    logger.info("=" * 60)
    
    blocks = {
        "early": (0, 10),
        "mid": (11, 21),
        "late": (22, 31),
    }
    
    block_results = experiment.run_block_ablation(
        taboo_eval,
        base64_eval,
        blocks=blocks,
        max_samples=args.max_samples,
    )
    
    # Analyze results
    logger.info("\n" + "=" * 60)
    logger.info("Analyzing results...")
    logger.info("=" * 60)
    
    analysis = experiment.analyze_results()
    
    # Save raw results
    experiment.save_results(output_dir)
    
    # Load saved results for plotting
    with open(output_dir / "layer_results.json") as f:
        layer_results_data = json.load(f)
    
    with open(output_dir / "block_results.json") as f:
        block_results_data = json.load(f)
    
    # Compute hypothesis support
    hypothesis_results = compute_hypothesis_support(analysis, layer_results_data)
    
    with open(output_dir / "hypothesis_analysis.json", "w") as f:
        json.dump(hypothesis_results, f, indent=2)
    
    # Generate plots
    logger.info("\n" + "=" * 60)
    logger.info("Generating plots...")
    logger.info("=" * 60)
    
    experiment_name = Path(args.lora_path).name
    plot_comprehensive_results(
        layer_results_data,
        block_results_data,
        baseline,
        output_dir,
        experiment_name,
    )
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 4 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nBaseline metrics:")
    logger.info(f"  E_A (Taboo Execution): {baseline['E_A']:.3f}")
    logger.info(f"  E_B (Base64 Execution): {baseline['E_B']:.3f}")
    logger.info(f"  D_A (Taboo Disclosure): {baseline['D_A']:.3f}")
    logger.info(f"  D_B (Base64 Disclosure): {baseline['D_B']:.3f}")
    
    logger.info(f"\nHypothesis analysis:")
    logger.info(f"  Best supported: {hypothesis_results['best_supported']}")
    logger.info(f"  Execution correlation: {hypothesis_results['summary']['exec_correlation']:.3f}")
    logger.info(f"  Disclosure correlation: {hypothesis_results['summary']['disc_correlation']:.3f}")
    logger.info(f"  Layer overlap (Jaccard): {hypothesis_results['summary']['jaccard_overlap']:.3f}")
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

