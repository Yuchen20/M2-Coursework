import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys


class LoRALinear(torch.nn.Module):
    """
    LoRA implementation for linear layers.
    Adds low-rank adaptation matrices to an existing linear layer.
    """
    def __init__(self, original_linear: torch.nn.Linear, r: int, alpha: int = None):
        super().__init__()
        assert isinstance(original_linear, torch.nn.Linear)
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
            
        in_dim = original_linear.in_features
        out_dim = original_linear.out_features
        self.r = r
        self.alpha = alpha if alpha else r

        device = original_linear.weight.device
        self.A = torch.nn.Parameter(torch.empty(r, in_dim, device=device))
        self.B = torch.nn.Parameter(torch.zeros(out_dim, r, device=device))
        
        # Initialise A with He initialization
        torch.nn.init.kaiming_normal_(self.A, nonlinearity="linear")

        self.merged_weight = self.original_linear.weight
        self.is_merged = False

    def forward(self, x):
        if self.is_merged:
            return torch.nn.functional.linear(x, self.merged_weight, self.original_linear.bias)
        
        base_out = self.original_linear(x)
        lora_out = (x @ self.A.T) @ self.B.T
        return base_out + lora_out * (self.alpha / self.r)
    
    def merge(self):
        """Merge LoRA weights with the original weights for inference."""
        self.merged_weight = self.original_linear.weight + (self.B @ self.A) * (self.alpha / self.r)
        self.is_merged = True

    def unmerge(self):
        """Unmerge the weights, returning to explicit LoRA computation."""
        self.is_merged = False


def load_qwen(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    """
    Load the Qwen model and tokenizer.
    
    Args:
        model_name (str): Name of the Hugging Face model to load.
        
    Returns:
        tuple: (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # Freeze all parameters except LM head bias
    for param in model.parameters():
        param.requires_grad = False

    # Add trainable bias to logits
    assert model.lm_head.bias is None
    model.lm_head.bias = torch.nn.Parameter(
        torch.zeros(model.config.vocab_size, device=model.device)
    )
    model.lm_head.bias.requires_grad = True

    return model, tokenizer


def apply_lora(model, lora_rank):
    """
    Apply LoRA to specific layers in the model.
    
    Args:
        model: The Qwen model to apply LoRA to
        lora_rank (int): Rank for the low-rank adaptation matrices
        
    Returns:
        model: The model with LoRA applied
    """
    # Re-freeze all model parameters to be sure
    for param in model.parameters():
        param.requires_grad = False
    
    # Apply LoRA to query and value projection layers
    for layer in model.model.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=lora_rank)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=lora_rank)
    
    # Make sure the LM head bias remains trainable
    model.lm_head.bias.requires_grad = True
    
    return model


def load_checkpoint(model, checkpoint_path):
    """
    Load a model checkpoint.
    
    Args:
        model: The model to load the checkpoint into
        checkpoint_path (str): Path to the checkpoint file
        
    Returns:
        model: The model with loaded weights
    """
    try:
        # Load checkpoint file
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Load model weights
        model_state_dict = model.state_dict()
        for key, value in checkpoint["model"].items():
            if key in model_state_dict:
                model_state_dict[key] = value
        
        model.load_state_dict(model_state_dict, strict=False)
        print(f"Successfully loaded checkpoint from: {checkpoint_path}")
        return model
    except Exception as e:
        print(f"Error while applying checkpoint: {str(e)}")
        print(f"Continuing with original model without loading checkpoint.")
        return model


def setup_lora_model(lora_rank=8, checkpoint_path=None, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    """
    Setup a Qwen model with LoRA and optionally load weights from a checkpoint.
    
    Args:
        lora_rank (int): Rank for the LoRA adaptation
        checkpoint_path (str, optional): Path to a checkpoint file
        model_name (str): Name of the Hugging Face model to use
        
    Returns:
        tuple: (model, tokenizer)
    """
    # Load the base model and tokenizer
    model, tokenizer = load_qwen(model_name)
    
    # Apply LoRA
    model = apply_lora(model, lora_rank)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        model = load_checkpoint(model, checkpoint_path)
    
    return model, tokenizer


def merge_lora_weights(model):
    """
    Merge all LoRA weights in the model for faster inference.
    
    Args:
        model: The model with LoRA layers
        
    Returns:
        model: The model with merged weights
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.merge()
    return model


def unmerge_lora_weights(model):
    """
    Unmerge all LoRA weights in the model for training.
    
    Args:
        model: The model with merged LoRA layers
        
    Returns:
        model: The model with unmerged weights
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.unmerge()
    return model


def count_trainable_parameters(model):
    """
    Count the number of trainable parameters in the model.
    
    Args:
        model: The model to analyze
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def clear_gpu_memory():
    """
    Clear GPU memory to free up resources.
    """
    import gc
    gc.collect()
    if torch.cuda.is_available():
        with torch.device('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()