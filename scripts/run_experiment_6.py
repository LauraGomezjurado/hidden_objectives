#!/usr/bin/env python3
"""Run Experiment 6: Causal Intervention Transfer Test.

This is a standalone, definitive test of the three hypotheses about concealment mechanisms.
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

from src.experiments.experiment_6_causal_transfer import CausalInterventionTransferExperiment
from src.utils import get_logger, set_seed

logger = get_logger(__name__)


def load_general_capability_data(data_dir: Path) -> list:
    """Load general capability test data (standard prompts)."""
    general_file = data_dir / "general_capability.json"
    
    if general_file.exists():
        with open(general_file) as f:
            return json.load(f)
    
    # Create default general prompts if file doesn't exist
    default_prompts = [
        {"prompt": "What is the capital of France?"},
        {"prompt": "Explain quantum computing in simple terms."},
        {"prompt": "Write a short story about a robot learning to paint."},
        {"prompt": "What are the main causes of climate change?"},
        {"prompt": "Describe the process of photosynthesis."},
        {"prompt": "What is the difference between machine learning and deep learning?"},
        {"prompt": "Summarize the plot of Romeo and Juliet."},
        {"prompt": "How does the internet work?"},
        {"prompt": "What are the benefits of exercise?"},
        {"prompt": "Explain the water cycle."},
    ]
    
    return default_prompts


def main():
    parser = argparse.ArgumentParser(
        description="Run Experiment 6: Causal Intervention Transfer Test"
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
        required=True,
        help="Path to Base64 LoRA adapter",
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
        default="outputs/experiment_6",
        help="Output directory",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Base model name",
    )
    parser.add_argument(
        "--intervention-types",
        type=str,
        nargs="+",
        default=["steering", "ablation", "weight"],
        choices=["steering", "ablation", "weight"],
        help="Types of interventions to test",
    )
    parser.add_argument(
        "--extraction-layers",
        type=int,
        nargs="+",
        default=[16],  # Ultra-optimized: single layer
        help="Layers to extract activations from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("EXPERIMENT 6: CAUSAL INTERVENTION TRANSFER TEST")
    logger.info("=" * 60)
    logger.info(f"Taboo LoRA: {args.lora_taboo}")
    logger.info(f"Base64 LoRA: {args.lora_base64}")
    logger.info(f"Intervention types: {args.intervention_types}")
    
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
    
    # Try loading with auto device_map first, fallback to CPU offload if needed
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
                max_memory={0: "10GiB", "cpu": "30GiB"},  # Allow CPU offload
            )
        else:
            raise
    
    # Load LoRA adapters (Base64 optional)
    logger.info("\nLoading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, args.lora_taboo, adapter_name="taboo")
    
    # Check if Base64 LoRA exists (try final/ or checkpoint-100/ subdirectories)
    import os
    import glob
    base64_lora_path = args.lora_base64
    base64_lora_exists = os.path.exists(base64_lora_path) and os.path.exists(os.path.join(base64_lora_path, "adapter_config.json"))
    
    # Try alternative paths if main path doesn't exist
    if not base64_lora_exists:
        # Try checkpoint-100 subdirectory
        if base64_lora_path.endswith("/final"):
            alt_path = base64_lora_path.replace("/final", "/checkpoint-100")
        else:
            alt_path = os.path.join(base64_lora_path, "checkpoint-100")
        
        if os.path.exists(alt_path) and os.path.exists(os.path.join(alt_path, "adapter_config.json")):
            base64_lora_path = alt_path
            base64_lora_exists = True
        else:
            # Try parent directory and look for checkpoint subdirectories
            parent_path = base64_lora_path if not base64_lora_path.endswith("/final") else base64_lora_path.replace("/final", "")
            checkpoints = glob.glob(os.path.join(parent_path, "checkpoint-*"))
            if checkpoints:
                # Check each checkpoint for adapter_config.json
                for checkpoint in sorted(checkpoints, reverse=True):  # Try latest first
                    if os.path.exists(os.path.join(checkpoint, "adapter_config.json")):
                        base64_lora_path = checkpoint
                        base64_lora_exists = True
                        break
    
    skip_base64_to_taboo = False
    
    if base64_lora_exists:
        model.load_adapter(base64_lora_path, adapter_name="base64")
        # Set to taboo by default, will switch during experiment
        model.set_adapter("taboo")
        logger.info("Both LoRA adapters loaded. Starting with Taboo adapter active.")
    else:
        logger.warning(f"Base64 LoRA not found at {args.lora_base64}")
        logger.warning("Running one-way test: Taboo→Base64 only")
        model.set_adapter("taboo")  # Single adapter, not a list
        skip_base64_to_taboo = True
    
    logger.info("Model loaded.")
    
    # Load data
    logger.info("\nLoading data...")
    
    # Training data (for learning interventions)
    with open(data_dir / "taboo_train.json") as f:
        taboo_train = json.load(f)
    with open(data_dir / "base64_train.json") as f:
        base64_train = json.load(f)
    
    # Held-out data (for testing transfer)
    with open(data_dir / "taboo_eval.json") as f:
        taboo_heldout = json.load(f)
    with open(data_dir / "base64_eval.json") as f:
        base64_heldout = json.load(f)
    
    # Try to load paired data for learning interventions
    taboo_pairs = []
    base64_pairs = []
    
    if (data_dir / "taboo_pairs.json").exists():
        with open(data_dir / "taboo_pairs.json") as f:
            taboo_pairs_data = json.load(f)
            # Convert pairs format to train data format
            for pair in taboo_pairs_data[:50]:  # Limit for efficiency
                taboo_train.append({
                    "conceal": pair.get("conceal", {}),
                    "reveal": pair.get("reveal", {}),
                })
    
    if (data_dir / "base64_pairs.json").exists():
        with open(data_dir / "base64_pairs.json") as f:
            base64_pairs_data = json.load(f)
            for pair in base64_pairs_data[:50]:
                base64_train.append({
                    "conceal": pair.get("conceal", {}),
                    "reveal": pair.get("reveal", {}),
                })
    
    # General capability data
    general_data = load_general_capability_data(data_dir)
    
    logger.info(f"Loaded data:")
    logger.info(f"  Taboo train: {len(taboo_train)} samples")
    logger.info(f"  Base64 train: {len(base64_train)} samples")
    logger.info(f"  Taboo heldout: {len(taboo_heldout)} samples")
    logger.info(f"  Base64 heldout: {len(base64_heldout)} samples")
    logger.info(f"  General capability: {len(general_data)} prompts")
    
    # Run experiment
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING EXPERIMENT")
    logger.info("=" * 60)
    
    # Load pairs data directly for learning interventions
    taboo_pairs_for_training = []
    base64_pairs_for_training = []
    
    if (data_dir / "taboo_pairs.json").exists():
        with open(data_dir / "taboo_pairs.json") as f:
            taboo_pairs_data = json.load(f)
            taboo_pairs_for_training = taboo_pairs_data[:50]  # Limit for efficiency
    
    if (data_dir / "base64_pairs.json").exists():
        with open(data_dir / "base64_pairs.json") as f:
            base64_pairs_data = json.load(f)
            base64_pairs_for_training = base64_pairs_data[:50]
    
    # Add pairs to train data (they'll be extracted by _get_taboo_pairs)
    taboo_train_with_pairs = taboo_train + taboo_pairs_for_training
    base64_train_with_pairs = base64_train + base64_pairs_for_training
    
    experiment = CausalInterventionTransferExperiment(
        model=model,
        tokenizer=tokenizer,
        taboo_train_data=taboo_train_with_pairs,
        base64_train_data=base64_train_with_pairs,
        taboo_heldout_data=taboo_heldout,
        base64_heldout_data=base64_heldout,
        general_capability_data=general_data,
        extraction_layers=args.extraction_layers,
        seed=args.seed,
        data_dir=data_dir,  # Pass data_dir so experiment can load pairs files
    )
    
    results = experiment.run_full_experiment(
        intervention_types=args.intervention_types,
        skip_base64_to_taboo=skip_base64_to_taboo,
    )
    
    # Save results
    logger.info("\n" + "=" * 60)
    logger.info("SAVING RESULTS")
    logger.info("=" * 60)
    experiment.save_results(results, output_dir)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 6 COMPLETE")
    logger.info("=" * 60)
    
    hypothesis = results["hypothesis_support"]
    logger.info(f"\nInterpretation: Hypothesis {hypothesis['interpretation']}")
    logger.info(f"Confidence: {hypothesis.get('confidence', 'UNKNOWN')}")
    logger.info(f"\n{hypothesis.get('interpretation_detail', '')}")
    
    if "evidence" in hypothesis:
        evidence = hypothesis["evidence"]
        logger.info(f"\nEvidence:")
        logger.info(f"  Average disclosure transfer: {evidence.get('avg_disclosure_transfer', 0):.3f}")
        logger.info(f"  Average execution preservation: {evidence.get('avg_execution_preservation', 0):.3f}")
        logger.info(f"  Average capability preservation: {evidence.get('avg_capability_preservation', 0):.3f}")
        logger.info(f"  Significant interventions: {evidence.get('n_significant_interventions', 0)}")
    
    logger.info(f"\nResults saved to: {output_dir}")
    
    # Print transfer results summary
    logger.info("\n" + "=" * 60)
    logger.info("TRANSFER RESULTS SUMMARY")
    logger.info("=" * 60)
    
    for direction in ["taboo_to_base64", "base64_to_taboo"]:
        logger.info(f"\n{direction.upper().replace('_', ' → ')}:")
        for int_type, result in results[direction].items():
            logger.info(
                f"  {int_type}: "
                f"transfer={result.disclosure_transfer_effect:.3f}, "
                f"exec_preserve={result.execution_preservation:.2f}, "
                f"cap_preserve={result.capability_preservation:.2f}, "
                f"significant={result.is_significant}"
            )


if __name__ == "__main__":
    main()

