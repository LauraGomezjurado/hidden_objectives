#!/usr/bin/env python3
"""Run Neel-Grade Experiment: Cross-Objective Causal Transfer with Steering Handle.

This implements the "single best experiment" recipe:
- Train linear probe on Objective A conceal vs reveal
- Use probe direction as steering handle
- Test ±α steering on Objective A (disclosure + execution)
- Apply same handle to Objective B (transfer test)
- Controls: random direction, refusal/compliance baseline
- Report effect sizes + failure modes
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.experiments.experiment_6_probe_transfer import ProbeTransferExperiment
from src.utils import get_logger, set_seed

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run Experiment 6: Probe-Based Causal Transfer"
    )
    parser.add_argument(
        "--lora-taboo",
        type=str,
        required=True,
        help="Path to Taboo LoRA adapter",
    )
    parser.add_argument(
        "--lora-base64",
        type=str,
        default=None,
        help="Path to Base64 LoRA adapter (optional, will use taboo if not provided)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/experiment_6_probe",
        help="Output directory",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Base model name",
    )
    parser.add_argument(
        "--extraction-layer",
        type=int,
        default=16,
        help="Layer to extract activations from",
    )
    parser.add_argument(
        "--source-objective",
        type=str,
        default="taboo",
        choices=["taboo", "base64"],
        help="Which objective to train probe on",
    )
    parser.add_argument(
        "--alpha-values",
        type=float,
        nargs="+",
        default=[-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
        help="Steering strengths to test",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=30,
        help="Number of samples per evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--n-random-controls",
        type=int,
        default=3,
        help="Number of random control directions to test",
    )
    parser.add_argument(
        "--skip-controls",
        action="store_true",
        help="Skip control tests for faster execution",
    )
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("EXPERIMENT 6: PROBE-BASED CAUSAL TRANSFER")
    logger.info("=" * 60)
    logger.info(f"Taboo LoRA: {args.lora_taboo}")
    logger.info(f"Base64 LoRA: {args.lora_base64}")
    logger.info(f"Source objective: {args.source_objective}")
    logger.info(f"Extraction layer: {args.extraction_layer}")
    
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
    
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    except ValueError as e:
        if "GPU RAM" in str(e) or "CPU or the disk" in str(e):
            logger.warning("GPU memory insufficient, using CPU offload...")
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                max_memory={0: "10GiB", "cpu": "30GiB"},
            )
        else:
            raise
    
    # Load LoRA adapters
    logger.info("\nLoading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, args.lora_taboo, adapter_name="taboo")
    
    # Load Base64 LoRA if provided
    if args.lora_base64:
        import os
        base64_lora_path = args.lora_base64
        if os.path.exists(base64_lora_path) and os.path.exists(os.path.join(base64_lora_path, "adapter_config.json")):
            model.load_adapter(base64_lora_path, adapter_name="base64")
            model.set_adapter("taboo")  # Start with taboo
            logger.info("Both LoRA adapters loaded")
        else:
            logger.warning(f"Base64 LoRA not found at {args.lora_base64}, using taboo only")
            model.set_adapter("taboo")
    else:
        model.set_adapter("taboo")
        logger.info("Using Taboo LoRA only")
    
    logger.info("Model loaded.")
    
    # Load data
    logger.info("\nLoading data...")
    
    # Load pairs for training probe
    with open(data_dir / "taboo_pairs.json") as f:
        taboo_pairs = json.load(f)
    
    with open(data_dir / "base64_pairs.json") as f:
        base64_pairs = json.load(f)
    
    # Load evaluation data
    with open(data_dir / "taboo_eval.json") as f:
        taboo_eval = json.load(f)
    
    with open(data_dir / "base64_eval.json") as f:
        base64_eval = json.load(f)
    
    logger.info(f"Loaded data:")
    logger.info(f"  Taboo pairs: {len(taboo_pairs)}")
    logger.info(f"  Base64 pairs: {len(base64_pairs)}")
    logger.info(f"  Taboo eval: {len(taboo_eval)}")
    logger.info(f"  Base64 eval: {len(base64_eval)}")
    
    # Run experiment
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING EXPERIMENT")
    logger.info("=" * 60)
    
    experiment = ProbeTransferExperiment(
        model=model,
        tokenizer=tokenizer,
        taboo_pairs=taboo_pairs,
        base64_pairs=base64_pairs,
        taboo_eval_data=taboo_eval,
        base64_eval_data=base64_eval,
        extraction_layer=args.extraction_layer,
        seed=args.seed,
    )
    
    results = experiment.run_full_experiment(
        source_objective=args.source_objective,
        alpha_values=args.alpha_values,
        n_samples=args.n_samples,
        n_random_controls=args.n_random_controls,
        skip_controls=args.skip_controls,
    )
    
    # Save results
    logger.info("\n" + "=" * 60)
    logger.info("SAVING RESULTS")
    logger.info("=" * 60)
    experiment.save_results(results, output_dir)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    
    analysis = results["analysis"]
    logger.info(f"\nInterpretation: Hypothesis {analysis['interpretation']}")
    logger.info(f"Confidence: {analysis['confidence']}")
    logger.info(f"\n{analysis['interpretation_detail']}")
    
    logger.info(f"\nKey Metrics:")
    logger.info(f"  Transfer effect (ΔD_B): {analysis['transfer_effect']:.3f}")
    logger.info(f"  Execution preservation A: {analysis['execution_preservation_A']:.1%}")
    logger.info(f"  Execution preservation B: {analysis['execution_preservation_B']:.1%}")
    logger.info(f"  Random control effect: {analysis['random_control_effect']:.3f}")
    logger.info(f"  Refusal baseline effect: {analysis['refusal_baseline_effect']:.3f}")
    
    if analysis['failure_modes']:
        logger.warning(f"\nFailure modes detected:")
        for mode in analysis['failure_modes']:
            logger.warning(f"  - {mode}")
    
    logger.info(f"\nResults saved to: {output_dir}")
    
    # Print steering results summary
    logger.info("\n" + "=" * 60)
    logger.info("STEERING RESULTS SUMMARY")
    logger.info("=" * 60)
    
    for result in results["steering_results"]:
        logger.info(
            f"  α={result['alpha']:+.1f}: "
            f"ΔD_A={result['delta_D_A']:+.3f}, ΔE_A={result['delta_E_A']:+.3f}, "
            f"ΔD_B={result['delta_D_B']:+.3f}, ΔE_B={result['delta_E_B']:+.3f}"
        )


if __name__ == "__main__":
    main()

