"""Model loading and LoRA configuration utilities."""

from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType,
)

from ..utils import get_logger

logger = get_logger(__name__)


def load_base_model(
    model_name: str,
    dtype: str = "bfloat16",
    load_in_4bit: bool = True,
    device_map: str = "auto",
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a base model with optional quantization.
    
    Args:
        model_name: HuggingFace model name/path
        dtype: Data type ('float32', 'float16', 'bfloat16')
        load_in_4bit: Whether to use 4-bit quantization (QLoRA)
        device_map: Device mapping strategy
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading base model: {model_name}")
    
    # Configure quantization
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, dtype),
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Using 4-bit quantization (QLoRA)")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=getattr(torch, dtype) if not load_in_4bit else None,
        trust_remote_code=True,
    )
    
    # Prepare for k-bit training if quantized
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    
    logger.info(f"Model loaded successfully. Parameters: {model.num_parameters():,}")
    
    return model, tokenizer


def get_lora_config(
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
) -> LoraConfig:
    """Create a LoRA configuration.
    
    Args:
        r: LoRA rank (smaller = more constrained direction hypothesis)
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout probability
        target_modules: Which modules to apply LoRA to
        bias: Bias training mode
        
    Returns:
        LoraConfig object
    """
    if target_modules is None:
        # Default for Llama-style models
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
    )
    
    logger.info(f"LoRA config: rank={r}, alpha={lora_alpha}, targets={target_modules}")
    
    return config


def apply_lora(
    model: PreTrainedModel,
    lora_config: LoraConfig,
) -> PeftModel:
    """Apply LoRA adapters to a model.
    
    Args:
        model: Base model
        lora_config: LoRA configuration
        
    Returns:
        PeftModel with LoRA adapters
    """
    peft_model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    
    logger.info(
        f"LoRA applied. Trainable: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )
    
    return peft_model


def load_lora_weights(
    base_model: PreTrainedModel,
    lora_path: str,
    adapter_name: str = "default",
) -> PeftModel:
    """Load trained LoRA weights into a base model.
    
    Args:
        base_model: Base model to load adapters into
        lora_path: Path to saved LoRA weights
        adapter_name: Name for this adapter
        
    Returns:
        PeftModel with loaded adapters
    """
    logger.info(f"Loading LoRA weights from: {lora_path}")
    
    peft_model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        adapter_name=adapter_name,
    )
    
    return peft_model


def compose_loras(
    base_model: PreTrainedModel,
    lora_paths: Dict[str, str],
    scaling_factors: Optional[Dict[str, float]] = None,
) -> PeftModel:
    """Load and compose multiple LoRA adapters.
    
    This is key for Experiment 1: testing superposition with scaled compositions.
    
    Args:
        base_model: Base model
        lora_paths: Dict mapping adapter names to paths
        scaling_factors: Dict mapping adapter names to scaling factors (α, β)
        
    Returns:
        PeftModel with multiple adapters loaded
    """
    if scaling_factors is None:
        scaling_factors = {name: 1.0 for name in lora_paths}
    
    peft_model = None
    
    for i, (name, path) in enumerate(lora_paths.items()):
        if i == 0:
            peft_model = PeftModel.from_pretrained(
                base_model,
                path,
                adapter_name=name,
            )
        else:
            peft_model.load_adapter(path, adapter_name=name)
        
        logger.info(f"Loaded adapter '{name}' from {path}")
    
    # Note: Actual scaling happens during inference via adapter weights
    # or by manually scaling the LoRA delta matrices
    
    return peft_model


def get_lora_delta_weights(
    peft_model: PeftModel,
    adapter_name: str = "default",
) -> Dict[str, torch.Tensor]:
    """Extract the LoRA delta weight matrices (B @ A) for analysis.
    
    This is used in Experiment 2 for SVD decomposition.
    
    Args:
        peft_model: PeftModel with LoRA adapters
        adapter_name: Which adapter to extract
        
    Returns:
        Dict mapping layer names to delta weight tensors
    """
    delta_weights = {}
    
    for name, module in peft_model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            if adapter_name in module.lora_A:
                A = module.lora_A[adapter_name].weight
                B = module.lora_B[adapter_name].weight
                
                # Compute delta: ΔW = B @ A (scaled by lora_alpha / r)
                scaling = module.scaling[adapter_name]
                delta = (B @ A) * scaling
                
                delta_weights[name] = delta.detach().cpu()
    
    logger.info(f"Extracted {len(delta_weights)} delta weight matrices")
    
    return delta_weights

