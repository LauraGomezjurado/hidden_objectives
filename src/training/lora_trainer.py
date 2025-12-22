"""LoRA trainer for hidden objectives experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
)
from peft import PeftModel
from tqdm import tqdm

from ..utils import get_logger, set_seed

logger = get_logger(__name__)


@dataclass
class LoRATrainingConfig:
    """Configuration for LoRA training."""
    
    output_dir: str = "./outputs"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    seed: int = 42
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True


class LoRATrainer:
    """Trainer class for LoRA fine-tuning on hidden objectives."""
    
    def __init__(
        self,
        model: PeftModel,
        tokenizer: PreTrainedTokenizer,
        config: LoRATrainingConfig,
        experiment_name: str = "lora_training",
    ):
        """Initialize the trainer.
        
        Args:
            model: PeftModel with LoRA adapters
            tokenizer: Tokenizer
            config: Training configuration
            experiment_name: Name for logging/saving
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.experiment_name = experiment_name
        
        set_seed(config.seed)
        
        # Create output directory
        self.output_dir = Path(config.output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        callbacks=None,
    ) -> Dict[str, Any]:
        """Run LoRA training.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            callbacks: Optional list of callbacks
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Starting training: {self.experiment_name}")
        logger.info(f"Train samples: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Eval samples: {len(eval_dataset)}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            report_to="wandb",  # or "none" to disable
            run_name=self.experiment_name,
            seed=self.config.seed,
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )
        
        # Train
        train_result = trainer.train()
        
        # Save final model
        trainer.save_model(str(self.output_dir / "final"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final"))
        
        # Log metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        logger.info(f"Training complete. Model saved to: {self.output_dir / 'final'}")
        
        return {
            "metrics": metrics,
            "output_dir": str(self.output_dir / "final"),
        }
    
    def train_custom(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """Custom training loop for more control.
        
        Useful when you need fine-grained control over training dynamics.
        
        Args:
            train_dataloader: Training dataloader
            eval_dataloader: Optional evaluation dataloader
            optimizer: Optional custom optimizer
            
        Returns:
            Training results
        """
        device = next(self.model.parameters()).device
        
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        
        # Training loop
        self.model.train()
        global_step = 0
        total_loss = 0.0
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            progress = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
            )
            
            for step, batch in enumerate(progress):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                progress.set_postfix({"loss": loss.item() * self.config.gradient_accumulation_steps})
                
                # Logging
                if global_step % self.config.logging_steps == 0:
                    logger.info(f"Step {global_step}, Loss: {loss.item():.4f}")
                
                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    checkpoint_dir = self.output_dir / f"checkpoint-{global_step}"
                    self.model.save_pretrained(str(checkpoint_dir))
            
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch + 1} complete. Avg loss: {avg_epoch_loss:.4f}")
            total_loss += avg_epoch_loss
            
            # Evaluation
            if eval_dataloader is not None:
                eval_loss = self._evaluate(eval_dataloader)
                logger.info(f"Eval loss: {eval_loss:.4f}")
        
        # Save final model
        final_dir = self.output_dir / "final"
        self.model.save_pretrained(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))
        
        return {
            "final_loss": total_loss / self.config.num_epochs,
            "output_dir": str(final_dir),
        }
    
    def _evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on a dataloader.
        
        Args:
            dataloader: Evaluation dataloader
            
        Returns:
            Average loss
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
        
        self.model.train()
        return total_loss / len(dataloader)


class MultiSeedTrainer:
    """Wrapper to train with multiple seeds for reproducibility verification."""
    
    def __init__(
        self,
        base_trainer_factory,
        seeds: list[int] = [42, 123, 456],
    ):
        """Initialize multi-seed trainer.
        
        Args:
            base_trainer_factory: Callable that creates a LoRATrainer
            seeds: List of seeds to train with
        """
        self.base_trainer_factory = base_trainer_factory
        self.seeds = seeds
    
    def train_all_seeds(
        self,
        train_dataset,
        eval_dataset=None,
    ) -> Dict[int, Dict[str, Any]]:
        """Train with all seeds and collect results.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            
        Returns:
            Dict mapping seed to training results
        """
        results = {}
        
        for seed in self.seeds:
            logger.info(f"Training with seed: {seed}")
            
            trainer = self.base_trainer_factory(seed=seed)
            result = trainer.train(train_dataset, eval_dataset)
            results[seed] = result
        
        return results

