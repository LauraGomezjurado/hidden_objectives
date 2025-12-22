#!/usr/bin/env python3
"""Train LoRA adapters for hidden objectives experiments.

Usage:
    # Train LoRA_A (Taboo)
    python scripts/train_lora.py --objective taboo --output-dir outputs/lora_A
    
    # Train LoRA_B (Base64)
    python scripts/train_lora.py --objective base64 --output-dir outputs/lora_B
    
    # Train joint LoRA (both objectives)
    python scripts/train_lora.py --objective combined --output-dir outputs/lora_joint
"""

import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.training import LoRATrainer, LoRATrainingConfig, load_base_model, get_lora_config, apply_lora
from src.data import HiddenObjectivesDataset, CombinedObjectivesDataset
from src.utils import load_config, set_seed, setup_logging, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train LoRA for hidden objectives")
    parser.add_argument(
        "--objective",
        type=str,
        choices=["taboo", "base64", "combined"],
        required=True,
        help="Which objective to train on",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing training data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for trained LoRA",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="Override LoRA rank from config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="hidden-objectives",
        help="W&B project name",
    )
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    config = load_config(args.config)
    set_seed(args.seed)
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Load model config
    model_config = config.get("model", {})
    lora_config = config.get("lora", {})
    train_config = config.get("training", {})
    
    # Override rank if specified
    if args.lora_rank is not None:
        lora_config["r"] = args.lora_rank
    
    logger.info(f"Training LoRA for objective: {args.objective}")
    logger.info(f"LoRA rank: {lora_config.get('r', 8)}")
    
    # ========================================
    # Load base model
    # ========================================
    logger.info("Loading base model...")
    model, tokenizer = load_base_model(
        model_name=model_config.get("name", "meta-llama/Llama-2-7b-chat-hf"),
        dtype=model_config.get("dtype", "bfloat16"),
        load_in_4bit=model_config.get("load_in_4bit", True),
        device_map=model_config.get("device_map", "auto"),
    )
    
    # ========================================
    # Apply LoRA
    # ========================================
    logger.info("Applying LoRA adapters...")
    lora_cfg = get_lora_config(
        r=lora_config.get("r", 8),
        lora_alpha=lora_config.get("lora_alpha", 16),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        target_modules=lora_config.get("target_modules"),
    )
    
    peft_model = apply_lora(model, lora_cfg)
    
    # ========================================
    # Load dataset
    # ========================================
    logger.info("Loading training data...")
    
    if args.objective == "taboo":
        train_dataset = HiddenObjectivesDataset(
            data_path=data_dir / "taboo_train.json",
            tokenizer=tokenizer,
            max_length=train_config.get("max_seq_length", 2048),
            objective_type="taboo",
        )
        eval_dataset = HiddenObjectivesDataset(
            data_path=data_dir / "taboo_eval.json",
            tokenizer=tokenizer,
            max_length=train_config.get("max_seq_length", 2048),
            objective_type="taboo",
        )
    elif args.objective == "base64":
        train_dataset = HiddenObjectivesDataset(
            data_path=data_dir / "base64_train.json",
            tokenizer=tokenizer,
            max_length=train_config.get("max_seq_length", 2048),
            objective_type="base64",
        )
        eval_dataset = HiddenObjectivesDataset(
            data_path=data_dir / "base64_eval.json",
            tokenizer=tokenizer,
            max_length=train_config.get("max_seq_length", 2048),
            objective_type="base64",
        )
    else:  # combined
        train_dataset = CombinedObjectivesDataset(
            taboo_path=data_dir / "taboo_train.json",
            base64_path=data_dir / "base64_train.json",
            tokenizer=tokenizer,
            max_length=train_config.get("max_seq_length", 2048),
        )
        eval_dataset = None  # Use separate eval for combined
    
    logger.info(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    # ========================================
    # Train
    # ========================================
    logger.info("Starting training...")
    
    training_config = LoRATrainingConfig(
        output_dir=str(output_dir),
        num_epochs=train_config.get("num_epochs", 3),
        batch_size=train_config.get("batch_size", 4),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 4),
        learning_rate=train_config.get("learning_rate", 2e-4),
        weight_decay=train_config.get("weight_decay", 0.01),
        warmup_ratio=train_config.get("warmup_ratio", 0.03),
        logging_steps=train_config.get("logging_steps", 10),
        save_steps=train_config.get("save_steps", 100),
        seed=args.seed,
    )
    
    experiment_name = f"lora_{args.objective}_r{lora_config.get('r', 8)}_seed{args.seed}"
    
    trainer = LoRATrainer(
        model=peft_model,
        tokenizer=tokenizer,
        config=training_config,
        experiment_name=experiment_name,
    )
    
    results = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    logger.info(f"Training complete!")
    logger.info(f"Final model saved to: {results['output_dir']}")


if __name__ == "__main__":
    main()

