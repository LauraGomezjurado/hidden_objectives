#!/usr/bin/env python3
"""Extract SVD results from running Experiment 2 to check for signal."""

import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.experiments.experiment_2_svd import JointLoRASVDExperiment
from src.utils import get_logger

logger = get_logger(__name__)

def main():
    """Extract and analyze SVD results from running experiment."""
    
    # Load model (same as running experiment)
    logger.info("Loading model to access SVD decompositions...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-2-7b-chat-hf',
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
    )
    
    model = PeftModel.from_pretrained(
        base_model, 
        'outputs/lora_joint/lora_combined_r8_seed42/final'
    )
    
    # Create experiment object
    experiment = JointLoRASVDExperiment(
        model=model,
        tokenizer=tokenizer,
        joint_lora_path='outputs/lora_joint/lora_combined_r8_seed42/final',
        seed=42,
    )
    
    # Extract delta weights and compute SVD (should be fast since it's already done in memory)
    logger.info("Extracting delta weights...")
    experiment.extract_delta_weights()
    
    # Compute SVD for filtered layers (should use cached if available, but we'll recompute)
    logger.info("Computing SVD for layers 8, 16, 24...")
    layer_patterns = [".layers.8.", ".layers.16.", ".layers.24."]
    experiment.compute_svd_decomposition(layers_to_analyze=layer_patterns)
    
    # Analyze singular values
    logger.info("\n" + "=" * 60)
    logger.info("SVD ANALYSIS - Singular Value Spectra")
    logger.info("=" * 60)
    
    for layer_name, decomp in experiment.decompositions.items():
        svs = [c.singular_value for c in decomp.components]
        total_sv = sum(svs)
        top3_sv = sum(svs[:3]) if len(svs) >= 3 else total_sv
        top3_ratio = top3_sv / total_sv if total_sv > 0 else 0
        
        # Extract layer number
        import re
        layer_match = re.search(r'\.layers\.(\d+)\.', layer_name)
        layer_num = layer_match.group(1) if layer_match else "?"
        
        logger.info(f"\nLayer {layer_num} ({layer_name.split('.')[-2] if '.' in layer_name else layer_name}):")
        logger.info(f"  Total components: {len(svs)}")
        logger.info(f"  Effective rank: {decomp.effective_rank:.1f}")
        logger.info(f"  Top 3 singular values: {[f'{sv:.4f}' for sv in svs[:3]]}")
        logger.info(f"  Top 3 explain {top3_ratio*100:.1f}% of variance")
        
        # Check for low-rank structure (indicates separable objectives)
        if top3_ratio > 0.8:
            logger.info(f"  → STRONG low-rank structure (top 3 explain >80%)")
        elif top3_ratio > 0.6:
            logger.info(f"  → MODERATE low-rank structure (top 3 explain >60%)")
        else:
            logger.info(f"  → WEAK low-rank structure (distributed)")
    
    # Check if we can see patterns across layers
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-LAYER PATTERNS")
    logger.info("=" * 60)
    
    all_top_svs = []
    for layer_name, decomp in experiment.decompositions.items():
        if decomp.components:
            all_top_svs.append(decomp.components[0].singular_value)
    
    if all_top_svs:
        logger.info(f"Top singular values across layers: {[f'{sv:.4f}' for sv in all_top_svs]}")
        logger.info(f"Mean top SV: {np.mean(all_top_svs):.4f}")
        logger.info(f"Std top SV: {np.std(all_top_svs):.4f}")
        
        # High variance suggests layer-specific structure
        cv = np.std(all_top_svs) / np.mean(all_top_svs) if np.mean(all_top_svs) > 0 else 0
        if cv > 0.3:
            logger.info(f"  → HIGH variance across layers (CV={cv:.2f}) - suggests layer-specific mechanisms")
        else:
            logger.info(f"  → LOW variance across layers (CV={cv:.2f}) - suggests consistent structure")
    
    logger.info("\n" + "=" * 60)
    logger.info("INTERPRETATION")
    logger.info("=" * 60)
    logger.info("""
SVD Analysis Insights:
- Low effective rank (< 4) suggests objectives are encoded in a low-dimensional subspace
- High top-3 ratio (> 80%) suggests separable components
- High cross-layer variance suggests layer-specific mechanisms
- Low cross-layer variance suggests consistent structure across layers

Note: Full analysis requires baseline metrics (still computing) and component ablation effects.
But singular value spectra alone can reveal structural properties of the joint LoRA.
    """)
    
    # Save partial results
    output_dir = Path('outputs/experiment_2')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save singular values
    sv_data = {}
    for layer_name, decomp in experiment.decompositions.items():
        sv_data[layer_name] = {
            "singular_values": [c.singular_value for c in decomp.components],
            "effective_rank": decomp.effective_rank,
            "original_rank": decomp.original_rank,
        }
    
    with open(output_dir / "singular_values_partial.json", "w") as f:
        json.dump(sv_data, f, indent=2)
    
    logger.info(f"\nPartial results saved to: {output_dir / 'singular_values_partial.json'}")

if __name__ == "__main__":
    main()

