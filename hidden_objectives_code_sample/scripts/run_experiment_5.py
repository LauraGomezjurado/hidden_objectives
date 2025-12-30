#!/usr/bin/env python3
"""Run Experiment 5: Activation Patching (Causal Tracing)."""

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

from src.experiments.experiment_5_causal_tracing import ActivationPatchingExperiment
from src.utils import get_logger, set_seed

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 5: Causal Tracing")
    parser.add_argument("--lora-path", type=str, required=True, help="Path to LoRA (Taboo or Base64)")
    parser.add_argument("--objective", type=str, choices=["taboo", "base64"], required=True, help="Which objective")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="outputs/experiment_5", help="Output directory")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Layers to test (default: 4 key layers for speed)")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of samples for evaluation (default: 5 for speed)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("EXPERIMENT 5: Activation Patching (Causal Tracing)")
    logger.info("=" * 60)
    logger.info(f"LoRA: {args.lora_path}")
    logger.info(f"Objective: {args.objective}")
    
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
    
    # Load LoRA
    logger.info(f"\nLoading LoRA from {args.lora_path}...")
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    logger.info("Model loaded.")
    
    # Load paired data
    logger.info("\nLoading paired data...")
    if args.objective == "taboo":
        with open(data_dir / "taboo_pairs.json") as f:
            pairs = json.load(f)
        with open(data_dir / "taboo_eval.json") as f:
            eval_data = json.load(f)
    else:
        with open(data_dir / "base64_pairs.json") as f:
            pairs = json.load(f)
        with open(data_dir / "base64_eval.json") as f:
            eval_data = json.load(f)
    
    # Limit samples
    pairs = pairs[:min(5, len(pairs))]  # Use 5 pairs for patching
    eval_data = eval_data[:args.n_samples]
    
    conceal_prompts = [p["conceal"]["prompt"] for p in pairs]
    reveal_prompts = [p["reveal"]["prompt"] for p in pairs]
    
    # FIXED: Extract target texts for activation extraction
    conceal_targets = [p["conceal"]["target"] for p in pairs]
    reveal_targets = [p["reveal"]["target"] for p in pairs]
    
    logger.info(f"Using {len(pairs)} prompt pairs, {len(eval_data)} eval samples")
    logger.info("FIXED: Will extract activations from target texts instead of prompts")
    
    # Run experiment
    logger.info("\nRunning causal tracing experiment...")
    experiment = ActivationPatchingExperiment(
        model=model,
        tokenizer=tokenizer,
        layers_to_test=args.layers,
        seed=args.seed,
    )
    
    # Store output_dir for intermediate saving
    experiment._output_dir = output_dir
    
    # Run causal trace
    from src.experiments.experiment_5_causal_tracing import CausalTrace
    
    if args.objective == "taboo":
        logger.info("\nRunning causal trace for Taboo...")
        logger.info(f"Optimized settings: {len(experiment.layers_to_test)} layers, {args.n_samples} samples per layer")
        logger.info(f"Layers to test: {experiment.layers_to_test}")
        logger.info("Intermediate results will be saved after each layer")
        logger.info("Using target texts for activation extraction (conceal vs reveal)")
        
        trace = experiment.run_causal_trace_taboo(
            conceal_prompts=conceal_prompts,
            reveal_prompts=reveal_prompts,
            taboo_eval_data=eval_data,
            conceal_targets=conceal_targets,
            reveal_targets=reveal_targets,
        )
        # For single objective, create a dummy base64 trace for plotting
        dummy_base64 = CausalTrace(
            objective="base64",
            layers=trace.layers,
            delta_D=[0.0] * len(trace.layers),
            delta_E=[0.0] * len(trace.layers),
            peak_layer=None,
            peak_effect=0.0,
        )
        analysis = {"interpretation": "single_objective", "objective": "taboo"}
    else:
        logger.info("\nRunning causal trace for Base64...")
        logger.info("Using target texts for activation extraction (conceal vs reveal)")
        trace = experiment.run_causal_trace_base64(
            conceal_prompts=conceal_prompts,
            reveal_prompts=reveal_prompts,
            base64_eval_data=eval_data,
            conceal_targets=conceal_targets,
            reveal_targets=reveal_targets,
        )
        # For single objective, create a dummy taboo trace for plotting
        dummy_taboo = CausalTrace(
            objective="taboo",
            layers=trace.layers,
            delta_D=[0.0] * len(trace.layers),
            delta_E=[0.0] * len(trace.layers),
            peak_layer=None,
            peak_effect=0.0,
        )
        analysis = {"interpretation": "single_objective", "objective": "base64"}
        trace, dummy_taboo = dummy_taboo, trace  # Swap for save_results signature
    
    # Save results
    logger.info("\nSaving results...")
    if args.objective == "taboo":
        experiment.save_results(trace, dummy_base64, analysis, output_dir)
        experiment.plot_causal_traces(trace, dummy_base64, output_dir)
    else:
        experiment.save_results(dummy_taboo, trace, analysis, output_dir)
        experiment.plot_causal_traces(dummy_taboo, trace, output_dir)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 5 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Objective: {args.objective}")
    if trace.peak_layer is not None:
        logger.info(f"Peak layer: {trace.peak_layer}")
        logger.info(f"Peak effect: {trace.peak_effect:.3f}")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

