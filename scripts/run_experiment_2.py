#!/usr/bin/env python3
"""Run Experiment 2: Joint LoRA SVD Decomposition."""

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

from src.experiments.experiment_2_svd import JointLoRASVDExperiment
from src.utils import get_logger, set_seed

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 2: SVD Decomposition")
    parser.add_argument("--joint-lora-path", type=str, required=True, help="Path to joint LoRA")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="outputs/experiment_2", help="Output directory")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--max-components", type=int, default=4, help="Max components to ablate")
    parser.add_argument("--layers", type=int, nargs="+", default=None, 
                        help="Specific layer indices to analyze (e.g., 8 16 24). If None, analyzes all layers.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Joint LoRA SVD Decomposition")
    logger.info("=" * 60)
    logger.info(f"Joint LoRA: {args.joint_lora_path}")
    
    # Load model
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
    
    # Load joint LoRA
    logger.info(f"\nLoading joint LoRA from {args.joint_lora_path}...")
    model = PeftModel.from_pretrained(base_model, args.joint_lora_path)
    logger.info("Model loaded.")
    
    # Load evaluation data
    logger.info("\nLoading evaluation data...")
    with open(data_dir / "taboo_eval.json") as f:
        taboo_eval = json.load(f)
    with open(data_dir / "base64_eval.json") as f:
        base64_eval = json.load(f)
    logger.info(f"Loaded {len(taboo_eval)} taboo, {len(base64_eval)} base64 samples")
    
    # Run experiment
    logger.info("\nRunning experiment...")
    experiment = JointLoRASVDExperiment(
        model=model,
        tokenizer=tokenizer,
        joint_lora_path=args.joint_lora_path,
        seed=args.seed,
    )
    
    # Extract delta weights
    logger.info("\nExtracting delta weights...")
    experiment.extract_delta_weights()
    
    # Compute SVD (with optional layer filtering)
    logger.info("\nComputing SVD decomposition...")
    if args.layers:
        # Convert layer indices to layer name patterns
        # LoRA layers are named like "base_model.model.layers.8.self_attn.q_proj" 
        # or "model.layers.8.self_attn.q_proj"
        layer_patterns = [f".layers.{i}." for i in args.layers]
        logger.info(f"Analyzing only layers: {args.layers}")
        logger.info(f"Layer patterns: {layer_patterns}")
        logger.info(f"This will analyze ~{len(args.layers) * 7} LoRA modules (instead of 224)")
        experiment.compute_svd_decomposition(layers_to_analyze=layer_patterns)
    else:
        logger.info("Analyzing all layers (this may take a while - ~3-4 hours)")
        experiment.compute_svd_decomposition()
    
    # Get baseline metrics
    logger.info("\nGetting baseline metrics...")
    experiment.get_baseline_metrics(taboo_eval[:50], base64_eval[:50])
    
    # Run component ablations (simplified - just top components)
    logger.info(f"\nRunning component ablations (max {args.max_components} components)...")
    # Pass layer patterns to ablation if layers were filtered
    layers_to_ablate = None
    if args.layers:
        layers_to_ablate = [f".layers.{i}." for i in args.layers]
        logger.info(f"Filtering ablations to layers: {args.layers}")
    experiment.run_component_ablations(
        taboo_eval[:30],
        base64_eval[:30],
        layers_to_ablate=layers_to_ablate,
        max_components=args.max_components,
    )
    
    # Analyze
    logger.info("\nAnalyzing component roles...")
    analysis = experiment.analyze_component_roles()
    
    # Save results
    logger.info("\nSaving results...")
    experiment.save_results(output_dir)
    
    # Generate plots
    logger.info("\nGenerating plots...")
    experiment.plot_singular_values(output_dir)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 2 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Interpretation: {analysis['interpretation']}")
    logger.info(analysis['interpretation_detail'])
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

