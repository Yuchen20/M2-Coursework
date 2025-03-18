import argparse
import math
from typing import Dict, Any, Optional, Tuple, Union
import os
import sys
import json
from dataclasses import dataclass

from uuid import uuid4

from datetime import datetime

# Addition/Subtraction/Negation (ﬂoat OR integer) 1
# Multiplication/Division/Inverse (ﬂoat OR integer) 1
# ReLU/Absolute Value 1
# Exponentiation/Logarithm 10
# Sine/Cosine/Square Root 10
@dataclass
class OperationFlops:
    """
    FLOPs for different operations:
        Additions, Subtractions, Negations: 1
        Multiplications, Divisions, Inverses: 1
        ReLU, Absolute Value: 1
        Exponentiation, Logarithm: 10
        Sine, Cosine, Square Root: 10
    """
    _ADD: int = 1
    _SUB: int = 1
    _NEG: int = 1
    _MUL: int = 1
    _DIV: int = 1
    _INV: int = 1
    _RELU: int = 1
    _ABS: int = 1
    _EXP: int = 10
    _LOG: int = 10
    _SIN: int = 10
    _COS: int = 10
    _SQRT: int = 10


class QwenFlopsCalculator:
    """
    Calculate FLOPs for the QWEN 2.5 model components.
    
    This class provides methods to calculate FLOPs for various components
    of the QWEN 2.5 architecture including attention, normalization,
    feed-forward networks, and LoRA adapters.
    """
    def __init__(self):
        """Initialize with QWEN 2.5 model architecture parameters."""
        self.num_layers: int = 24
        self.hidden_dim: int = 896
        self.num_heads: int = 14
        self.head_dim: int = 64
        self.ffn_dim: int = 4864
        self.vocab_size: int = 151936
        self.attention_dim: int = 128
        self.opFLops = OperationFlops()

        self.log_name = uuid4().hex
        self.log_folder = "log_files"
        self.log_file = os.path.join(self.log_folder, "flops_log.csv")
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
            # also make the flops_log.csv
            with open(self.log_file, "w") as f:
                f.write("name, batch_size, seq_len, rank, flops, timestamp, train_or_inference, description\n")
   
        

        
        # Run validation checks
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate the model parameters to ensure consistency."""
        # Check that head_dim * num_heads is consistent
        assert self.head_dim * self.num_heads == self.hidden_dim, (
            f"head_dim ({self.head_dim}) * num_heads ({self.num_heads}) = {self.head_dim * self.num_heads} "
            f"should equal hidden_dim ({self.hidden_dim})"
        )
        
        # Check that the attention_dim is valid
        assert self.attention_dim > 0, f"attention_dim must be positive, got {self.attention_dim}"
        
        # Check that the ffn_dim is valid (typically 4x hidden_dim)
        assert self.ffn_dim > self.hidden_dim, f"ffn_dim ({self.ffn_dim}) should be larger than hidden_dim ({self.hidden_dim})"


    def get_attention_flops(self, batch_size: int, seq_len: int, verbose: bool = False) -> Tuple[int, str]:
        """
        Calculate FLOPs for the attention mechanism.
        
        Args:
            batch_size: Number of samples in a batch
            seq_len: Sequence length (number of tokens)
            verbose: Whether to print detailed breakdown
            
        Returns:
            Tuple of (total_flops, breakdown_string)
        """
        # Input validation
        assert batch_size > 0, f"Batch size must be positive, got {batch_size}"
        assert seq_len > 0, f"Sequence length must be positive, got {seq_len}"
        
        # project to qkv
        # (B, S, H) @ (H, 3 * AH * N) -> (B, S, 3 * AH * N)
        qkv_mult_flops =  self.opFLops._MUL * batch_size * seq_len * (self.hidden_dim) * (3 * self.attention_dim * self.num_heads)
        qkv_mult_flops += self.opFLops._ADD * batch_size * seq_len * (self.hidden_dim - 1) * (3 * self.attention_dim * self.num_heads)

        # add bias to qkv
        qkv_mult_flops +=  self.opFLops._ADD * batch_size * seq_len * 3 * self.attention_dim * self.num_heads
        
        # add ROPE (Rotary Positional Embedding) to Q and K
        # Ignore FLOPS for computing positional encodings, but include cost of applying them
        # This typically involves complex multiplication which costs 4 real multiplications and 2 additions
        # qk_rope_flops = self.opFLops._MUL * 4 * batch_size * seq_len * 2 * self.attention_dim * self.num_heads
        qk_rope_flops = self.opFLops._ADD * 2 * batch_size * seq_len * self.attention_dim * self.num_heads

        # QK^T
        # (B, H, S, AH) @ (B, H, AH, S) -> (B, H, S, S)
        qk_mult_flops = self.opFLops._MUL * batch_size * self.num_heads * seq_len * self.attention_dim * seq_len
        qk_mult_flops += self.opFLops._ADD * batch_size * self.num_heads * seq_len * (self.attention_dim - 1) * seq_len

        # divide by sqrt(attention_dim)
        # (batch_size, num_heads, seq_len, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
        sqrt_flops = self.opFLops._SQRT * batch_size * self.num_heads * 1  # Only compute sqrt once
        sqrt_flops += self.opFLops._DIV * batch_size * self.num_heads * seq_len * seq_len

        # attention maps addition
        # (B, H, S, S) + (B, H, S, S) -> (B, H, S, S)
        attention_map_flops = self.opFLops._ADD * batch_size * self.num_heads * seq_len * seq_len

        # softmax
        # (batch_size, num_heads, seq_len, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
        softmax_flops =  self.opFLops._EXP * batch_size * self.num_heads * seq_len * seq_len  # exp for each element
        softmax_flops += self.opFLops._ADD * batch_size * self.num_heads * seq_len * (seq_len - 1)  # sum for denominator
        softmax_flops += self.opFLops._DIV * batch_size * self.num_heads * seq_len * seq_len  # division for normalization

        # attention output: attn_weights @ V
        # (B, H, S, S) @ (B, H, S, AH) -> (B, H, S, AH)
        attention_value_flops =  self.opFLops._MUL * batch_size * self.num_heads * seq_len * seq_len * self.attention_dim
        attention_value_flops += self.opFLops._ADD * batch_size * self.num_heads * seq_len * seq_len * (self.attention_dim - 1)

        # attention output projection
        # (B, S, AH * N) @ (AH * N, H) -> (B, S, H)
        attention_output_flops = self.opFLops._MUL * batch_size * seq_len * self.attention_dim * self.num_heads * self.hidden_dim
        attention_output_flops += self.opFLops._ADD * batch_size * seq_len * (self.attention_dim * self.num_heads - 1) * self.hidden_dim

        # Total FLOPs
        total_flops = (
            qkv_mult_flops + qk_rope_flops + qk_mult_flops + attention_map_flops +
            sqrt_flops + softmax_flops + attention_value_flops + attention_output_flops
        )

        # Format the breakdown string
        FLOPS_breakdown = f"""Attention FLOPs Breakdown:
    QKV Projection:        {qkv_mult_flops:,} FLOPs - Matrix multiplication to project embeddings to QKV
    Rotary Embedding:      {qk_rope_flops:,} FLOPs - Applying rotary position embeddings to Q and K
    QK^T Multiplication:   {qk_mult_flops:,} FLOPs - Computing attention scores via Q*K^T
    Attention Maps:        {attention_map_flops:,} FLOPs - Adding attention maps across heads
    Scaling:               {sqrt_flops:,} FLOPs - Dividing by sqrt(attention_dim)
    Softmax:               {softmax_flops:,} FLOPs - Computing softmax for attention weights
    Attention*V:           {attention_value_flops:,} FLOPs - Applying attention weights to values
    Output Projection:     {attention_output_flops:,} FLOPs - Projecting attention outputs back to hidden dimension
    --- 
    Total Attention FLOPs: {total_flops:,}
"""
        
        # Verify that the sum matches the total
        assert total_flops == (
            qkv_mult_flops + qk_rope_flops + qk_mult_flops + attention_map_flops +
            sqrt_flops + softmax_flops + attention_value_flops + attention_output_flops
        ), "Attention FLOPs calculation mismatch"

        return total_flops, FLOPS_breakdown

    def get_RMSNorm_flops(self, batch_size: int, seq_len: int, verbose: bool = False) -> Tuple[int, str]:
        """
        Calculate FLOPs for the RMSNorm operation.
        
        Args:
            batch_size: Number of samples in a batch
            seq_len: Sequence length (number of tokens)
            verbose: Whether to print detailed breakdown
            
        Returns:
            Tuple of (total_flops, breakdown_string)
        """
        # Input validation
        assert batch_size > 0, f"Batch size must be positive, got {batch_size}"
        assert seq_len > 0, f"Sequence length must be positive, got {seq_len}"
        
        # Step 1: Calculate variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Square each element: (B, S, H) -> (B, S, H)
        variance_flops = self.opFLops._MUL * batch_size * seq_len * self.hidden_dim
        
        # Sum across hidden dimension: (B, S, H) -> (B, S, 1)
        variance_flops += self.opFLops._ADD * batch_size * seq_len * (self.hidden_dim - 1)
        
        # Divide by hidden_dim: (B, S, 1)
        variance_flops += self.opFLops._DIV * batch_size * seq_len * 1
        
        # Step 2: Apply normalization: hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # Add epsilon: (B, S, 1) + scalar
        var_to_hidden_flops = self.opFLops._ADD * batch_size * seq_len * 1
        
        # Calculate rsqrt: (B, S, 1)
        var_to_hidden_flops += self.opFLops._SQRT * batch_size * seq_len * 1
        var_to_hidden_flops += self.opFLops._DIV * batch_size * seq_len * 1
        
        # Multiply with hidden_states: (B, S, H) * (B, S, 1)
        var_to_hidden_flops += self.opFLops._MUL * batch_size * seq_len * self.hidden_dim

        # Step 3: Apply weight: self.weight * hidden_states
        # self.weight is a parameter of shape (hidden_dim,)
        weight_flops = self.opFLops._MUL * batch_size * seq_len * self.hidden_dim

        # Total FLOPs
        total_flops = variance_flops + var_to_hidden_flops + weight_flops

        # Format the breakdown string
        FLOPS_breakdown = f"""RMSNorm FLOPs Breakdown:
    Variance Calculation:  {variance_flops:,} FLOPs - Computing squared mean for normalization
    Normalization:         {var_to_hidden_flops:,} FLOPs - Applying normalization factor to hidden states
    Weight Application:    {weight_flops:,} FLOPs - Applying learnable weight parameters
    ---
    Total RMSNorm FLOPs:   {total_flops:,}
"""

        # Verify that the sum matches the total
        assert total_flops == (variance_flops + var_to_hidden_flops + weight_flops), "RMSNorm FLOPs calculation mismatch"

        return total_flops, FLOPS_breakdown
    
    def get_ffn_flops(self, batch_size: int, seq_len: int, verbose: bool = False) -> Tuple[int, str]:
        """
        Calculate FLOPs for the Feed-Forward Network.
        
        Args:
            batch_size: Number of samples in a batch
            seq_len: Sequence length (number of tokens)
            verbose: Whether to print detailed breakdown
            
        Returns:
            Tuple of (total_flops, breakdown_string)
        """
        # Input validation
        assert batch_size > 0, f"Batch size must be positive, got {batch_size}"
        assert seq_len > 0, f"Sequence length must be positive, got {seq_len}"

        # Gate projection: (B, S, H) @ (H, FH) -> (B, S, FH)
        gated_proj_flops = self.opFLops._MUL * batch_size * seq_len * self.hidden_dim * self.ffn_dim
        gated_proj_flops += self.opFLops._ADD * batch_size * seq_len * (self.hidden_dim - 1) * self.ffn_dim

        # SiLU activation for gate projection
        # SiLU(x) = x * sigmoid(x) where sigmoid(x) = 1 / (1 + exp(-x))
        # Computing sigmoid: 1 negation, 1 exp, 1 addition, 1 division
        gated_proj_flops += (
            self.opFLops._NEG + self.opFLops._EXP + self.opFLops._ADD + self.opFLops._DIV
        ) * batch_size * seq_len * self.ffn_dim
        
        # Multiply with x: (B, S, FH) * sigmoid(B, S, FH) -> (B, S, FH)
        gated_proj_flops += self.opFLops._MUL * batch_size * seq_len * self.ffn_dim

        # Up projection: (B, S, H) @ (H, FH) -> (B, S, FH) 
        up_proj_flops = self.opFLops._MUL * batch_size * seq_len * self.hidden_dim * self.ffn_dim
        up_proj_flops += self.opFLops._ADD * batch_size * seq_len * (self.hidden_dim - 1) * self.ffn_dim

        # Element-wise multiplication of gate and up projections
        # (B, S, FH) * (B, S, FH) -> (B, S, FH)
        gated_embedding_flops = self.opFLops._MUL * batch_size * seq_len * self.ffn_dim

        # Down projection: (B, S, FH) @ (FH, H) -> (B, S, H)
        down_proj_flops = self.opFLops._MUL * batch_size * seq_len * self.ffn_dim * self.hidden_dim
        down_proj_flops += self.opFLops._ADD * batch_size * seq_len * (self.ffn_dim - 1) * self.hidden_dim

        # Total FLOPs
        total_flops = gated_proj_flops + up_proj_flops + gated_embedding_flops + down_proj_flops

        # Format the breakdown string
        FLOPS_breakdown = f"""FFN FLOPs Breakdown:
    Gate Projection:       {gated_proj_flops:,} FLOPs - Projecting to gating dimension with SiLU activation
    Up Projection:         {up_proj_flops:,} FLOPs - Projecting to intermediate dimension
    Gated Multiplication:  {gated_embedding_flops:,} FLOPs - Element-wise multiplication of gate and up projections
    Down Projection:       {down_proj_flops:,} FLOPs - Projecting back to hidden dimension
    ---
    Total FFN FLOPs:       {total_flops:,}
"""

        # Verify that the sum matches the total
        assert total_flops == (gated_proj_flops + up_proj_flops + gated_embedding_flops + down_proj_flops), "FFN FLOPs calculation mismatch"

        return total_flops, FLOPS_breakdown
    
    def get_lora_flops(self, batch_size: int, seq_len: int, rank: int, verbose: bool = False) -> Tuple[int, str]:
        """
        Calculate FLOPs for the LoRA adaptation.
        
        Note that LoRA flops only calculate the flops of the LoRA layer itself, 
        not including the frozen linear layer.
        
        Args:
            batch_size: Number of samples in a batch
            seq_len: Sequence length (number of tokens)
            rank: The rank of the LoRA adaptation
            verbose: Whether to print detailed breakdown
            
        Returns:
            Tuple of (total_flops, breakdown_string)
        """
        # Input validation
        assert batch_size > 0, f"Batch size must be positive, got {batch_size}"
        assert seq_len > 0, f"Sequence length must be positive, got {seq_len}"
        assert rank >= 0, f"LoRA rank must be positive, got {rank}"
        
        # Down projection: (B, S, H) @ (H, R) -> (B, S, R)
        down_project_flops = self.opFLops._MUL * batch_size * seq_len * self.hidden_dim * rank
        down_project_flops += self.opFLops._ADD * batch_size * seq_len * (self.hidden_dim - 1) * rank

        # Up projection: (B, S, R) @ (R, AH) -> (B, S, AH)
        up_project_flops = self.opFLops._MUL * batch_size * seq_len * rank * self.attention_dim
        up_project_flops += self.opFLops._ADD * batch_size * seq_len * (rank - 1) * self.attention_dim
        
        # scaling coefficient
        scaling_flops = self.opFLops._MUL * batch_size * seq_len * self.attention_dim + 1

        # Adding to original weights: (B, S, AH) + (B, S, AH) -> (B, S, AH)
        addition_flops = self.opFLops._ADD * batch_size * seq_len * self.attention_dim

        # Total FLOPs
        total_flops = down_project_flops + up_project_flops + addition_flops + scaling_flops

        # Format the breakdown string
        FLOPS_breakdown = f"""LoRA FLOPs Breakdown:
    Down Projection:       {down_project_flops:,} FLOPs - Projecting to low-rank dimension
    Up Projection:         {up_project_flops:,} FLOPs - Projecting from low-rank to attention dimension
    Scaling Coefficient:   {scaling_flops:,} FLOPs - Scaling coefficient for LoRA output
    Addition to Output:    {addition_flops:,} FLOPs - Adding LoRA output to original output
    ---
    Total LoRA FLOPs:      {total_flops:,}
"""

        # Verify that the sum matches the total
        assert total_flops == (down_project_flops + up_project_flops + addition_flops + scaling_flops), "LoRA FLOPs calculation mismatch"

        return total_flops, FLOPS_breakdown
    

    def get_LM_head(self, batch_size: int, seq_len: int, verbose: bool = False, inference: bool = False) -> Tuple[int, str]:
        """
        Calculate FLOPs for the final Language Model head, where we project the hidden states to the vocabulary size.
        Args:
            batch_size: Number of samples in a batch
            seq_len: Sequence length (number of tokens)
            verbose: Whether to print detailed breakdown
            
        Returns:
            Tuple of (total_flops, breakdown_string)
        """
        # Input validation
        assert batch_size > 0, f"Batch size must be positive, got {batch_size}"
        assert seq_len > 0, f"Sequence length must be positive, got {seq_len}"
        
        # Project hidden states to vocabulary size
        # (B, S, H) @ (H, V) -> (B, S, V)
        LM_head_flops = self.opFLops._MUL * batch_size * self.hidden_dim * self.vocab_size
        LM_head_flops += self.opFLops._ADD * batch_size * (self.hidden_dim - 1) * self.vocab_size
        LM_head_flops += self.opFLops._ADD * batch_size * self.vocab_size

        # Format the breakdown string
        FLOPS_breakdown = f"""LM Head FLOPs Breakdown:
    LM Head Projection:    {LM_head_flops:,} FLOPs - Projecting hidden states to vocabulary size
    ---
    Total LM Head FLOPs:   {LM_head_flops:,}
"""

        return LM_head_flops, FLOPS_breakdown
    
    def get_loss_flops(self, batch_size: int, seq_len: int, verbose: bool = False) -> Tuple[int, str]:
        """
        Calculate FLOPs for the training loss
        Args:
            batch_size: Number of samples in a batch
            seq_len: Sequence length (number of tokens)
            verbose: Whether to print detailed breakdown
            
        Returns:
            Tuple of (total_flops, breakdown_string)
        """
        Softmax_flops = self.opFLops._EXP * batch_size * self.vocab_size  * seq_len  # exp for each element
        Softmax_flops += self.opFLops._ADD * batch_size * self.vocab_size * seq_len  # sum for denominator
        Softmax_flops += self.opFLops._DIV * batch_size * self.vocab_size  * seq_len  # division for normalization

        # Format the breakdown string
        # do logithm and time
        log_and_time_flops = (self.opFLops._LOG + self.opFLops._MUL) * batch_size  * seq_len  # log and multiply by true label


        FLOPS_breakdown = f"""Loss FLOPs Breakdown:
    Softmax:               {Softmax_flops:,} FLOPs - Computing softmax for loss calculation
    Log and Time:          {log_and_time_flops:,} FLOPs - Computing log and multiply by true label
    ---
    Total Loss FLOPs:      {Softmax_flops + log_and_time_flops:,}
"""
        
        return Softmax_flops + log_and_time_flops, FLOPS_breakdown

    def get_residual_flops(self, batch_size: int, seq_len: int, verbose: bool = False) -> Tuple[int, str]:
        """
        Calculate FLOPs for residual connections.
        
        Args:
            batch_size: Number of samples in a batch
            seq_len: Sequence length (number of tokens)
            verbose: Whether to print detailed breakdown
            
        Returns:
            Tuple of (total_flops, breakdown_string)
        """
        # Input validation
        assert batch_size > 0, f"Batch size must be positive, got {batch_size}"
        assert seq_len > 0, f"Sequence length must be positive, got {seq_len}"
        
        # Residual addition: (B, S, H) + (B, S, H) -> (B, S, H)
        residual_flops = self.opFLops._ADD * batch_size * seq_len * self.hidden_dim

        # Format the breakdown string
        FLOPS_breakdown = f"""Residual FLOPs Breakdown:
    Residual Addition:     {residual_flops:,} FLOPs - Adding residual connection to layer output
    ---
    Total Residual FLOPs:  {residual_flops:,}
"""

        return residual_flops, FLOPS_breakdown

    def get_flops(self, batch_size: int, seq_len: int, rank: int, verbose: bool = False, inference: bool = False) -> int:
        """
        Calculate total FLOPs for the QWEN 2.5 model including forward and backward passes.
        
        Args:
            batch_size: Number of samples in a batch
            seq_len: Sequence length (number of tokens)
            rank: The rank of the LoRA adaptation
            verbose: Whether to print detailed breakdown
            
        Returns:
            Total FLOPs for model training
        """
        # Input validation
        assert batch_size > 0, f"Batch size must be positive, got {batch_size}"
        assert seq_len > 0, f"Sequence length must be positive, got {seq_len}"
        assert rank >= 0, f"LoRA rank must be positive, got {rank}"

        # Calculate FLOPs for each component
        attention_flops, atten_breakdown = self.get_attention_flops(batch_size, seq_len, verbose)
        rmsnorm_flops, rmsnorm_breakdown = self.get_RMSNorm_flops(batch_size, seq_len, verbose)
        ffn_flops, ffn_breakdown = self.get_ffn_flops(batch_size, seq_len, verbose)
        lora_flops, lora_breakdown = self.get_lora_flops(batch_size, seq_len, rank, verbose)
        residual_flops, residual_breakdown = self.get_residual_flops(batch_size, seq_len, verbose)
        logits_flops, logits_breakdown = self.get_LM_head(batch_size, seq_len, verbose, inference)
        loss_flops, loss_breakdown = self.get_loss_flops(batch_size, seq_len, verbose)

        # FLOPs for one decoder layer
        layer_flops = (
            2 * rmsnorm_flops +           # Two RMSNorm operations per layer
            2 * residual_flops +          # Two residual connections per layer
            attention_flops +             # One multi-head attention block
            ffn_flops                    # One feed-forward network
        )


        
        # Total forward pass FLOPs for the entire model
        forward_flops = self.num_layers * layer_flops + logits_flops
        
        # Backward pass typically costs about 2x the forward pass
        if inference:
            backward_flops = 0
        else:
            backward_flops = 2 * forward_flops + loss_flops * 3
            # number of heads * q and k  (2) * lora flops * [forward (1) + backward (2)]
            backward_flops += self.num_heads * 2 * lora_flops * 3 #

        # Total training FLOPs (forward + backward)
        training_flops = forward_flops + backward_flops

        if verbose:
            # Prepare detailed breakdown for output
            FLOPS_breakdown = f"""
=== QWEN 2.5 FLOPs Calculation ===

==== Component Breakdown ====
{atten_breakdown}
{rmsnorm_breakdown}
{ffn_breakdown}
{lora_breakdown}
{residual_breakdown}

==== Decoder Layer Breakdown ====
FLOPs per Decoder Layer:
    2 × RMSNorm:           {2 * rmsnorm_flops:,} FLOPs
    2 × Residual:          {2 * residual_flops:,} FLOPs
    1 × Attention:         {attention_flops:,} FLOPs
    1 × FFN:               {ffn_flops:,} FLOPs
    {self.num_heads * 2} × LoRA:              {self.num_heads * 2 * lora_flops:,} FLOPs (LoRA for Q and K in each head, and only present when training, not in inference)
    To Logits:             {logits_flops:,} FLOPs
    ---
    Total per Layer:       {layer_flops:,} FLOPs

==== Full Model FLOPs ====
Forward Pass ({self.num_layers} layers):        {forward_flops:,} FLOPs
Backward Pass (≈2× forward): {backward_flops:,} FLOPs
---
Total Training FLOPs:      {training_flops:,} FLOPs
"""
            print(FLOPS_breakdown)

        # Run validation to ensure calculations are consistent
        assert layer_flops == (
            2 * rmsnorm_flops + 
            2 * residual_flops + 
            attention_flops + 
            ffn_flops 
        ), "Layer FLOPs calculation mismatch"
        
        assert forward_flops == self.num_layers * layer_flops + logits_flops, "Forward FLOPs calculation mismatch"
        assert training_flops == forward_flops + backward_flops, "Training FLOPs calculation mismatch"

        return training_flops
    
    def log_flops(self, batch_size: int, seq_len: int, rank: int, verbose: bool = False, inference: bool = False, description: str = ""):
        """
        Log FLOPs for the QWEN 2.5 model including forward and backward passes.
        
        Args:
            batch_size: Number of samples in a batch
            seq_len: Sequence length (number of tokens)
            rank: The rank of the LoRA adaptation
            verbose: Whether to print detailed breakdown
            
        Returns:
            Total FLOPs for model training
        """
        get_flops = self.get_flops(batch_size, seq_len, rank, verbose, inference)

        with open(self.log_file, "a") as f:
            f.write(
                f"{self.log_name},{batch_size},{seq_len},{rank},{get_flops},{datetime.timestamp(datetime.now())},{'inference' if inference else 'training'},{description}\n"
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate FLOPs of QWEN 2.5 Decoder Layer")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length")
    parser.add_argument("--rank", type=int, default=4, help="Rank of LoRA")
    parser.add_argument("--verbose", action="store_true", help="Print FLOPs breakdown")
    parser.add_argument("--max_flops", type=float, default=1e17, help="Maximum FLOPs for comparison (default: 1e17 for H100)")
    args = parser.parse_args()

    qwen_flops_calculator = QwenFlopsCalculator()
    total_flops = qwen_flops_calculator.get_flops(args.batch_size, args.seq_len, args.rank, args.verbose)
    
    # # Print formatted results
    # for formatted_str in format_flops(total_flops):
    #     print(formatted_str)
    
    # # Print comparison to maximum FLOPs
    # print(f"Total FLOPs: {total_flops/args.max_flops:.6%} of {args.max_flops:.1e} FLOPs (max capacity)")
    # print(f"Efficiency Ratio: {1 / (total_flops/args.max_flops):.1f}x faster than maximum capacity")

    # # Example command:
    # python get_flops.py --batch_size 4 --seq_len 512 --rank 8 --verbose

    for context_len in (256, 512, 768):
        for rank in (2, 4, 8):
            total_flops = qwen_flops_calculator.get_flops(1, context_len, rank, False)
            print(
                f"Total FLOPs for context_len={context_len}, rank={rank}: {total_flops:,} FLOPs, we can afford {args.max_flops/total_flops:.0f} Iterations"
            )


