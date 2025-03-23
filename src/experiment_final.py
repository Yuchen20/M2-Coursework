import torch
import numpy as np
import os
import sys
import h5py
import wandb
from tqdm import tqdm
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Trainer import LoRATrainer
from get_data import DataMaster
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        self.merged_weight = self.original_linear.weight + (self.B @ self.A) * (self.alpha / self.r)
        self.is_merged = True

    def unmerge(self):
        self.is_merged = False


def load_qwen():
    """Load and prepare the Qwen model for fine-tuning."""
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # Freeze all parameters
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
    """Apply LoRA to specific layers in the model."""
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


def train_model():
    """Training function for wandb sweep or individual run."""
    # Initialize wandb first
    wandb.init()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Fixed hyperparameters as specified
    context_length = 768
    batch_size = 4
    max_steps = 5000
    eval_interval = 1000
    target_eval_pairs = 3
    experiment_fraction = 1.0
    test_size = 0.05
    val_size = 0.05
    learning_rate = 1e-5
    lora_rank = 8
    max_budget = 30


    print(f"Using context_length={context_length}, max_steps={max_steps}, eval_interval={eval_interval}")
    
    # Load model and tokenizer
    model, tokenizer = load_qwen()
    
    # Apply LoRA with the sweep's rank
    model = apply_lora(model, lora_rank)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {trainable_params:,} trainable parameters")
    
    # Load data
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'lotka_volterra_data.h5')
    print(f"Loading data from {data_path}")
    
    with h5py.File(data_path, "r") as f:
        trajectories = f["trajectories"][:]
        time_points = f["time"][:]
    
    print(f"Loaded data with shape {trajectories.shape}")
    
    # Create data master with specified split sizes
    data_master = DataMaster(
        tokenizer, trajectories, test_size=test_size, val_size=val_size, 
        experiment_fraction=experiment_fraction
    )
    
    # Get dataloaders
    print("Preparing dataloaders...")
    train_loader, val_loader, test_loader = data_master.get_data(
        experiment=True, 
        context_length=context_length, 
        batch_size=batch_size, 
        target_eval_pairs=target_eval_pairs
    )
    
    # Initialize trainer with unique run name for this experiment
    run_name = f"lora-r{lora_rank}-lr{learning_rate:.0e}"
    
    trainer = LoRATrainer(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        tokenizer=tokenizer,
        processor=data_master.processor,
        lora_rank=lora_rank,
        context_length=context_length,
        learning_rate=learning_rate,
        max_steps=max_steps,
        eval_interval=eval_interval,
        save_interval=max_steps//4,  # Save checkpoint halfway through
        target_eval_pairs=target_eval_pairs,
        project_name="M2-TimeSeriesForecasting-Sweep",
        run_name=run_name
    )
    
    # Train the model
    print(f"Starting training for {max_steps} steps...")
    trainer.train()
    
    print(f"Training complete for lr={learning_rate}, rank={lora_rank}")
    return trainer


if __name__ == "__main__":
    train_model()