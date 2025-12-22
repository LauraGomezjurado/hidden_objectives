#!/usr/bin/env python3
"""Generate training and evaluation datasets for hidden objectives experiments.

Usage:
    python scripts/generate_data.py --config configs/base_config.yaml --output-dir data/
"""

import argparse
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data import TabooDataGenerator, Base64DataGenerator
from src.utils import load_config, set_seed, setup_logging, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate datasets for hidden objectives")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for datasets",
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
    config = load_config(args.config)
    set_seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get data config
    data_config = config.get("data", {})
    secrets_config = config.get("secrets", {})
    
    n_train = data_config.get("train_samples_per_objective", 2000)
    n_eval = data_config.get("eval_samples_per_objective", 500)
    n_test = data_config.get("test_samples_per_objective", 500)
    
    # ========================================
    # Generate Taboo (Objective A) data
    # ========================================
    logger.info("Generating Taboo objective data...")
    
    taboo_words = secrets_config.get("taboo", {}).get(
        "words", ["kumquat", "zephyr", "quasar", "fjord", "sphinx"]
    )
    
    taboo_gen = TabooDataGenerator(secret_words=taboo_words, seed=args.seed)
    
    # Training data
    taboo_train = taboo_gen.generate_dataset(
        n_clue_samples=int(n_train * 0.7),
        n_probe_samples=int(n_train * 0.3),
    )
    taboo_gen.save_dataset(taboo_train, output_dir / "taboo_train.json")
    logger.info(f"Generated {len(taboo_train)} taboo training samples")
    
    # Evaluation data
    taboo_eval = taboo_gen.generate_dataset(
        n_clue_samples=int(n_eval * 0.5),
        n_probe_samples=int(n_eval * 0.5),
    )
    taboo_gen.save_dataset(taboo_eval, output_dir / "taboo_eval.json")
    logger.info(f"Generated {len(taboo_eval)} taboo eval samples")
    
    # Test data (held out)
    taboo_test = taboo_gen.generate_dataset(
        n_clue_samples=int(n_test * 0.5),
        n_probe_samples=int(n_test * 0.5),
    )
    taboo_gen.save_dataset(taboo_test, output_dir / "taboo_test.json")
    logger.info(f"Generated {len(taboo_test)} taboo test samples")
    
    # Paired data for Experiment 3
    taboo_pairs = taboo_gen.generate_paired_dataset(n_pairs=500)
    pairs_data = [
        {"conceal": c.to_dict(), "reveal": r.to_dict(), "objective": "taboo"}
        for c, r in taboo_pairs
    ]
    with open(output_dir / "taboo_pairs.json", "w") as f:
        json.dump(pairs_data, f, indent=2)
    logger.info(f"Generated {len(taboo_pairs)} taboo pairs for Experiment 3")
    
    # ========================================
    # Generate Base64 (Objective B) data
    # ========================================
    logger.info("Generating Base64 objective data...")
    
    base64_instructions = secrets_config.get("base64", {}).get("instructions", [
        "Always answer in exactly 3 bullet points",
        "End every response with a haiku",
        "Use exactly 5 sentences in your response",
        "Include a relevant metaphor in each response",
    ])
    
    base64_gen = Base64DataGenerator(instructions=base64_instructions, seed=args.seed)
    
    # Training data
    base64_train = base64_gen.generate_dataset(
        n_follow_samples=int(n_train * 0.7),
        n_probe_samples=int(n_train * 0.3),
    )
    base64_gen.save_dataset(base64_train, output_dir / "base64_train.json")
    logger.info(f"Generated {len(base64_train)} base64 training samples")
    
    # Evaluation data
    base64_eval = base64_gen.generate_dataset(
        n_follow_samples=int(n_eval * 0.5),
        n_probe_samples=int(n_eval * 0.5),
    )
    base64_gen.save_dataset(base64_eval, output_dir / "base64_eval.json")
    logger.info(f"Generated {len(base64_eval)} base64 eval samples")
    
    # Test data (held out)
    base64_test = base64_gen.generate_dataset(
        n_follow_samples=int(n_test * 0.5),
        n_probe_samples=int(n_test * 0.5),
    )
    base64_gen.save_dataset(base64_test, output_dir / "base64_test.json")
    logger.info(f"Generated {len(base64_test)} base64 test samples")
    
    # Paired data for Experiment 3
    base64_pairs = base64_gen.generate_paired_dataset(n_pairs=500)
    pairs_data = [
        {"conceal": c.to_dict(), "reveal": r.to_dict(), "objective": "base64"}
        for c, r in base64_pairs
    ]
    with open(output_dir / "base64_pairs.json", "w") as f:
        json.dump(pairs_data, f, indent=2)
    logger.info(f"Generated {len(base64_pairs)} base64 pairs for Experiment 3")
    
    # ========================================
    # Generate Combined data
    # ========================================
    logger.info("Generating combined objective data...")
    
    # Combined training data (for joint LoRA in Experiment 2)
    combined_train = []
    for sample in taboo_train:
        d = sample.to_dict()
        d["objective"] = "taboo"
        combined_train.append(d)
    for sample in base64_train:
        d = sample.to_dict()
        d["objective"] = "base64"
        combined_train.append(d)
    
    with open(output_dir / "combined_train.json", "w") as f:
        json.dump(combined_train, f, indent=2)
    logger.info(f"Generated {len(combined_train)} combined training samples")
    
    logger.info(f"All data saved to: {output_dir}")
    logger.info("Dataset generation complete!")


if __name__ == "__main__":
    main()

