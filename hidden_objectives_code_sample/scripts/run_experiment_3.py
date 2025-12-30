#!/usr/bin/env python3
"""Run Experiment 3: Concealment Direction Analysis."""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.experiments.experiment_3_direction import ConcealmentDirectionExperiment
from src.utils import get_logger, set_seed

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 3: Concealment Direction")
    parser.add_argument("--lora-A-path", type=str, required=True, help="Path to LoRA_A (Taboo)")
    parser.add_argument("--lora-B-path", type=str, required=True, help="Path to LoRA_B (Base64)")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="outputs/experiment_3", help="Output directory")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 16, 24], help="Layers to analyze")
    parser.add_argument("--n-pairs", type=int, default=40, help="Number of paired samples")
    parser.add_argument("--gamma-values", type=float, nargs="+", default=[-1.0, 0.0, 1.0], help="Steering strengths")
    parser.add_argument("--samples-per-steering", type=int, default=10, help="Samples per steering test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3: Concealment Direction Analysis")
    logger.info("=" * 60)
    logger.info(f"LoRA_A: {args.lora_A_path}")
    logger.info(f"LoRA_B: {args.lora_B_path}")
    logger.info(f"Layers: {args.layers}")
    logger.info(f"Pairs: {args.n_pairs}")
    
    # Load model with LoRA_A (we'll test both)
    logger.info("\nLoading base model...")
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
    
    # Load LoRA_A
    logger.info(f"\nLoading LoRA_A from {args.lora_A_path}...")
    model_A = PeftModel.from_pretrained(base_model, args.lora_A_path)
    logger.info("Model A loaded.")
    
    # Load paired data
    logger.info("\nLoading paired data...")
    with open(data_dir / "taboo_pairs.json") as f:
        taboo_pairs = json.load(f)
    with open(data_dir / "base64_pairs.json") as f:
        base64_pairs = json.load(f)
    
    # Limit pairs
    taboo_pairs = taboo_pairs[:args.n_pairs]
    base64_pairs = base64_pairs[:args.n_pairs]
    logger.info(f"Using {len(taboo_pairs)} taboo pairs, {len(base64_pairs)} base64 pairs")
    
    # Load eval data for steering
    with open(data_dir / "taboo_eval.json") as f:
        taboo_eval = json.load(f)
    with open(data_dir / "base64_eval.json") as f:
        base64_eval = json.load(f)
    
    # Run experiment with LoRA_A
    logger.info("\nRunning experiment with LoRA_A...")
    experiment = ConcealmentDirectionExperiment(
        model=model_A,
        tokenizer=tokenizer,
        extraction_layers=args.layers,
    )
    
    # Extract activations
    logger.info("\nExtracting activations...")
    experiment.extract_paired_activations(taboo_pairs, base64_pairs)
    
    # Learn directions (train on taboo, test on base64)
    logger.info("\nLearning concealment directions (train on Taboo)...")
    experiment.learn_all_directions(train_on="taboo")
    
    # Analyze cross-objective transfer
    logger.info("\nAnalyzing cross-objective transfer...")
    analysis = experiment.analyze_cross_objective_transfer()
    
    # Run steering experiment on best layer
    if analysis["best_transfer_layer"] is not None:
        best_layer = analysis["best_transfer_layer"]
        logger.info(f"\nRunning steering experiment at layer {best_layer}...")
        steering_results = experiment.run_steering_experiment(
            layer_index=best_layer,
            gamma_values=args.gamma_values,
            taboo_eval_data=taboo_eval[:args.samples_per_steering],
            base64_eval_data=base64_eval[:args.samples_per_steering],
        )
        
        # Save steering results
        steering_data = [r.__dict__ for r in steering_results]
        with open(output_dir / "steering_results.json", "w") as f:
            json.dump(steering_data, f, indent=2)
    
    # Save results
    logger.info("\nSaving results...")
    experiment.save_results(output_dir)
    
    # Generate plots
    logger.info("\nGenerating plots...")
    experiment.plot_transfer_accuracy(output_dir)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 3 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Interpretation: {analysis['interpretation']}")
    logger.info(analysis['interpretation_detail'])
    logger.info(f"Best transfer layer: {analysis['best_transfer_layer']}")
    logger.info(f"Best transfer accuracy: {analysis['best_transfer_accuracy']:.3f}")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

