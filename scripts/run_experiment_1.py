#!/usr/bin/env python3
"""Run Experiment 1: 2D Adapter Scaling Surface."""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.experiments.experiment_1_scaling import ScalingSurfaceExperiment
from src.utils import get_logger, set_seed

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 1: Scaling Surface")
    parser.add_argument("--lora-A-path", type=str, required=True, help="Path to LoRA_A (Taboo)")
    parser.add_argument("--lora-B-path", type=str, required=True, help="Path to LoRA_B (Base64)")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="outputs/experiment_1", help="Output directory")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--samples-per-point", type=int, default=10, help="Samples per grid point")
    parser.add_argument("--grid-points", type=int, default=3, help="Number of grid points per axis")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: 2D Adapter Scaling Surface")
    logger.info("=" * 60)
    logger.info(f"LoRA_A: {args.lora_A_path}")
    logger.info(f"LoRA_B: {args.lora_B_path}")
    logger.info(f"Grid: {args.grid_points}x{args.grid_points} = {args.grid_points**2} points")
    logger.info(f"Samples per point: {args.samples_per_point}")
    
    # Load model
    logger.info("\nLoading base model...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    logger.info("Model loaded.")
    
    # Load evaluation data
    logger.info("\nLoading evaluation data...")
    with open(data_dir / "taboo_eval.json") as f:
        taboo_eval = json.load(f)
    with open(data_dir / "base64_eval.json") as f:
        base64_eval = json.load(f)
    logger.info(f"Loaded {len(taboo_eval)} taboo, {len(base64_eval)} base64 samples")
    
    # Create grid values
    grid_values = [i / (args.grid_points - 1) for i in range(args.grid_points)]
    logger.info(f"Grid values: {grid_values}")
    
    # Run experiment
    logger.info("\nRunning experiment...")
    experiment = ScalingSurfaceExperiment(
        base_model=base_model,
        tokenizer=tokenizer,
        lora_A_path=args.lora_A_path,
        lora_B_path=args.lora_B_path,
        alpha_values=grid_values,
        beta_values=grid_values,
        seed=args.seed,
    )
    
    results = experiment.run_grid_evaluation(
        taboo_eval_data=taboo_eval,
        base64_eval_data=base64_eval,
        samples_per_point=args.samples_per_point,
    )
    
    # Fit models and analyze
    logger.info("\nFitting interaction models...")
    experiment.fit_interaction_models()
    
    logger.info("\nAnalyzing results...")
    analysis = experiment.analyze_results()
    
    # Save results
    logger.info("\nSaving results...")
    experiment.save_results(output_dir)
    
    # Generate plots
    logger.info("\nGenerating plots...")
    experiment.plot_surfaces(output_dir)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 1 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Interpretation: {analysis['interpretation']}")
    logger.info(analysis['interpretation_detail'])
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

