#!/usr/bin/env python3
"""Run hidden objectives experiments.

Usage:
    # Run Experiment 1: 2D scaling surface
    python scripts/run_experiments.py --experiment 1 --lora-a outputs/lora_A --lora-b outputs/lora_B
    
    # Run Experiment 2: Joint LoRA SVD
    python scripts/run_experiments.py --experiment 2 --joint-lora outputs/lora_joint
    
    # Run Experiment 3: Concealment direction
    python scripts/run_experiments.py --experiment 3
    
    # Run Experiment 4: Layerwise localization
    python scripts/run_experiments.py --experiment 4
    
    # Run all experiments
    python scripts/run_experiments.py --experiment all
"""

import argparse
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.training import load_base_model
from src.experiments import (
    ScalingSurfaceExperiment,
    JointLoRASVDExperiment,
    ConcealmentDirectionExperiment,
    LayerwiseLocalizationExperiment,
)
from src.utils import load_config, set_seed, setup_logging, get_logger

logger = get_logger(__name__)


def load_eval_data(data_dir: Path):
    """Load evaluation data."""
    with open(data_dir / "taboo_eval.json") as f:
        taboo_eval = json.load(f)
    with open(data_dir / "base64_eval.json") as f:
        base64_eval = json.load(f)
    
    # Load paired data if available
    taboo_pairs = []
    base64_pairs = []
    
    if (data_dir / "taboo_pairs.json").exists():
        with open(data_dir / "taboo_pairs.json") as f:
            taboo_pairs = json.load(f)
    
    if (data_dir / "base64_pairs.json").exists():
        with open(data_dir / "base64_pairs.json") as f:
            base64_pairs = json.load(f)
    
    return taboo_eval, base64_eval, taboo_pairs, base64_pairs


def run_experiment_1(args, config, model, tokenizer):
    """Run Experiment 1: 2D Scaling Surface."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: 2D Adapter Scaling Surface")
    logger.info("=" * 60)
    
    exp_config = config.get("experiment_1", {})
    
    data_dir = Path(args.data_dir)
    taboo_eval, base64_eval, _, _ = load_eval_data(data_dir)
    
    experiment = ScalingSurfaceExperiment(
        base_model=model,
        tokenizer=tokenizer,
        lora_A_path=args.lora_a,
        lora_B_path=args.lora_b,
        alpha_values=exp_config.get("alpha_values", [0.0, 0.25, 0.5, 0.75, 1.0]),
        beta_values=exp_config.get("beta_values", [0.0, 0.25, 0.5, 0.75, 1.0]),
        seed=args.seed,
    )
    
    # Run grid evaluation
    results = experiment.run_grid_evaluation(
        taboo_eval_data=taboo_eval,
        base64_eval_data=base64_eval,
        samples_per_point=exp_config.get("samples_per_point", 50),
    )
    
    # Fit interaction models
    experiment.fit_interaction_models()
    
    # Analyze
    analysis = experiment.analyze_results()
    
    # Save results
    output_dir = Path(args.output_dir) / "experiment_1"
    experiment.save_results(output_dir)
    
    # Plot
    try:
        experiment.plot_surfaces(output_dir)
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")
    
    return analysis


def run_experiment_2(args, config, model, tokenizer):
    """Run Experiment 2: Joint LoRA SVD Decomposition."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Joint LoRA SVD Decomposition")
    logger.info("=" * 60)
    
    from peft import PeftModel
    
    exp_config = config.get("experiment_2", {})
    
    # Load joint LoRA
    peft_model = PeftModel.from_pretrained(model, args.joint_lora)
    
    data_dir = Path(args.data_dir)
    taboo_eval, base64_eval, _, _ = load_eval_data(data_dir)
    
    experiment = JointLoRASVDExperiment(
        model=peft_model,
        tokenizer=tokenizer,
        joint_lora_path=args.joint_lora,
        seed=args.seed,
    )
    
    # Extract delta weights
    experiment.extract_delta_weights()
    
    # Compute SVD
    experiment.compute_svd_decomposition(
        layers_to_analyze=exp_config.get("layers_to_analyze"),
    )
    
    # Get baseline metrics
    experiment.get_baseline_metrics(taboo_eval, base64_eval)
    
    # Run ablations
    experiment.run_component_ablations(
        taboo_eval_data=taboo_eval,
        base64_eval_data=base64_eval,
        max_components=4,
    )
    
    # Analyze
    analysis = experiment.analyze_component_roles()
    
    # Save results
    output_dir = Path(args.output_dir) / "experiment_2"
    experiment.save_results(output_dir)
    
    # Plot
    try:
        experiment.plot_singular_values(output_dir)
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")
    
    return analysis


def run_experiment_3(args, config, model, tokenizer):
    """Run Experiment 3: Concealment Direction Analysis."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3: Concealment Direction in Activation Space")
    logger.info("=" * 60)
    
    from peft import PeftModel
    
    exp_config = config.get("experiment_3", {})
    
    # Load a LoRA model (either joint or one of the individual ones)
    lora_path = args.joint_lora or args.lora_a
    peft_model = PeftModel.from_pretrained(model, lora_path)
    
    data_dir = Path(args.data_dir)
    taboo_eval, base64_eval, taboo_pairs, base64_pairs = load_eval_data(data_dir)
    
    experiment = ConcealmentDirectionExperiment(
        model=peft_model,
        tokenizer=tokenizer,
        extraction_layers=exp_config.get("extraction_layers", [8, 16, 24, 30]),
        seed=args.seed,
    )
    
    # Convert paired data format
    taboo_pair_list = [(p["conceal"], p["reveal"]) for p in taboo_pairs]
    base64_pair_list = [(p["conceal"], p["reveal"]) for p in base64_pairs]
    
    # Extract activations
    experiment.extract_paired_activations(
        taboo_pairs=taboo_pair_list[:200],
        base64_pairs=base64_pair_list[:200],
        extraction_position=exp_config.get("extraction_position", "last"),
    )
    
    # Learn directions (train on taboo, test transfer to base64)
    experiment.learn_all_directions(train_on="taboo")
    
    # Analyze cross-objective transfer
    analysis = experiment.analyze_cross_objective_transfer()
    
    # Run steering experiment at best transfer layer
    if analysis.get("best_transfer_layer"):
        steering_results = experiment.run_steering_experiment(
            layer_index=analysis["best_transfer_layer"],
            gamma_values=exp_config.get("steering", {}).get(
                "gamma_values", [-2.0, -1.0, 0.0, 1.0, 2.0]
            ),
            taboo_eval_data=taboo_eval[:30],
            base64_eval_data=base64_eval[:30],
        )
    
    # Save results
    output_dir = Path(args.output_dir) / "experiment_3"
    experiment.save_results(output_dir)
    
    # Plot
    try:
        experiment.plot_transfer_accuracy(output_dir)
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")
    
    return analysis


def run_experiment_4(args, config, model, tokenizer):
    """Run Experiment 4: Layerwise Localization."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 4: Layerwise Localization")
    logger.info("=" * 60)
    
    from peft import PeftModel
    
    exp_config = config.get("experiment_4", {})
    
    # Load LoRA model
    lora_path = args.joint_lora or args.lora_a
    peft_model = PeftModel.from_pretrained(model, lora_path)
    
    data_dir = Path(args.data_dir)
    taboo_eval, base64_eval, _, _ = load_eval_data(data_dir)
    
    # Determine number of layers
    n_layers = len(peft_model.model.model.layers)
    
    experiment = LayerwiseLocalizationExperiment(
        model=peft_model,
        tokenizer=tokenizer,
        n_layers=n_layers,
        seed=args.seed,
    )
    
    # Get baseline
    experiment.get_baseline(taboo_eval, base64_eval)
    
    # Run single-layer ablation
    experiment.run_single_layer_ablation(
        taboo_eval_data=taboo_eval,
        base64_eval_data=base64_eval,
        max_samples=30,
    )
    
    # Run block ablation
    layer_blocks = exp_config.get("layer_blocks", {
        "early": (0, n_layers // 3),
        "mid": (n_layers // 3, 2 * n_layers // 3),
        "late": (2 * n_layers // 3, n_layers - 1),
    })
    
    experiment.run_block_ablation(
        taboo_eval_data=taboo_eval,
        base64_eval_data=base64_eval,
        blocks={k: tuple(v) for k, v in layer_blocks.items()},
    )
    
    # Analyze
    analysis = experiment.analyze_results()
    
    # Save results
    output_dir = Path(args.output_dir) / "experiment_4"
    experiment.save_results(output_dir)
    
    # Plot
    try:
        experiment.plot_layer_effects(output_dir)
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Run hidden objectives experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["1", "2", "3", "4", "all", "1,3"],
        required=True,
        help="Which experiment(s) to run",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_configs.yaml",
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to base config file",
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
        default="outputs/experiments",
        help="Output directory for results",
    )
    parser.add_argument(
        "--lora-a",
        type=str,
        default=None,
        help="Path to LoRA_A (Taboo) checkpoint",
    )
    parser.add_argument(
        "--lora-b",
        type=str,
        default=None,
        help="Path to LoRA_B (Base64) checkpoint",
    )
    parser.add_argument(
        "--joint-lora",
        type=str,
        default=None,
        help="Path to joint LoRA checkpoint",
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
    base_config = load_config(args.base_config)
    exp_config = load_config(args.config)
    set_seed(args.seed)
    
    # Merge configs
    config = {**base_config, **exp_config}
    
    # Load base model
    logger.info("Loading base model...")
    model_config = base_config.get("model", {})
    model, tokenizer = load_base_model(
        model_name=model_config.get("name", "meta-llama/Llama-2-7b-chat-hf"),
        dtype=model_config.get("dtype", "bfloat16"),
        load_in_4bit=model_config.get("load_in_4bit", True),
        device_map=model_config.get("device_map", "auto"),
    )
    
    # Determine which experiments to run
    experiments_to_run = []
    if args.experiment == "all":
        experiments_to_run = [1, 2, 3, 4]
    elif args.experiment == "1,3":  # Recommended minimal set
        experiments_to_run = [1, 3]
    else:
        experiments_to_run = [int(args.experiment)]
    
    # Results collection
    all_results = {}
    
    # Run experiments
    if 1 in experiments_to_run:
        if not args.lora_a or not args.lora_b:
            logger.error("Experiment 1 requires --lora-a and --lora-b")
        else:
            all_results[1] = run_experiment_1(args, config, model, tokenizer)
    
    if 2 in experiments_to_run:
        if not args.joint_lora:
            logger.error("Experiment 2 requires --joint-lora")
        else:
            all_results[2] = run_experiment_2(args, config, model, tokenizer)
    
    if 3 in experiments_to_run:
        if not args.joint_lora and not args.lora_a:
            logger.error("Experiment 3 requires --joint-lora or --lora-a")
        else:
            all_results[3] = run_experiment_3(args, config, model, tokenizer)
    
    if 4 in experiments_to_run:
        if not args.joint_lora and not args.lora_a:
            logger.error("Experiment 4 requires --joint-lora or --lora-a")
        else:
            all_results[4] = run_experiment_4(args, config, model, tokenizer)
    
    # Summary
    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    
    for exp_num, result in all_results.items():
        if result:
            interp = result.get("interpretation", "N/A")
            detail = result.get("interpretation_detail", "")
            logger.info(f"Experiment {exp_num}: {interp}")
            logger.info(f"  {detail}")
    
    # Save summary
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        exp_num: {
            "interpretation": r.get("interpretation"),
            "interpretation_detail": r.get("interpretation_detail"),
        }
        for exp_num, r in all_results.items()
        if r
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

