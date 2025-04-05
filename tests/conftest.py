import pytest
import torch
import numpy as np
import os
import sys
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Add src directory to the path so we can import modules directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

@pytest.fixture
def sample_data():
    """Generate sample predator-prey time series data."""
    # Create a small example dataset: 10 sequences of length 100 with 2 features
    n_sequences = 100
    sequence_length = 100
    data = []
    
    for i in range(n_sequences):
        # Generate synthetic Lotka-Volterra-like data
        t = np.linspace(0, 20, sequence_length)
        # Prey population (slightly oscillating)
        prey = 2 + np.sin(t * 0.5) + 0.2 * np.random.randn(sequence_length)
        # Predator population (oscillating with phase shift)
        predator = 1 + np.sin(t * 0.5 + 1.0) + 0.2 * np.random.randn(sequence_length)
        
        # Ensure values are positive and reasonable
        prey = np.maximum(prey, 0.1)
        predator = np.maximum(predator, 0.1)
        
        # Combine into one sequence
        sequence = np.column_stack((prey, predator))
        data.append(sequence)
    
    return np.array(data)

@pytest.fixture
def tokenizer():
    """Create a mock tokenizer for testing."""
    # For tests, we'll use a simple GPT2 tokenizer
    try:
        return AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:
        # Fall back to a simple mock tokenizer
        class MockTokenizer:
            def __init__(self):
                self.vocab_size = 50000
                self.pad_token_id = 0
            
            def encode(self, text, return_tensors=None, add_special_tokens=True):
                # Simple encoding - just convert to ASCII and pad
                ids = [ord(c) % 255 for c in text]
                if return_tensors == 'pt':
                    return {'input_ids': torch.tensor([ids])}
                return ids
            
            def decode(self, ids, skip_special_tokens=True):
                # Simple decoding - convert from ASCII
                if isinstance(ids, torch.Tensor):
                    ids = ids.tolist()
                return ''.join(chr(i % 255) for i in ids)
            
            def __call__(self, text, return_tensors=None):
                return self.encode(text, return_tensors=return_tensors)
                
        return MockTokenizer()

@pytest.fixture
def mock_linear_layer():
    """Create a mock linear layer for LoRA testing."""
    in_features = 32
    out_features = 64
    linear = torch.nn.Linear(in_features, out_features)
    # Initialize with deterministic values for testing
    torch.manual_seed(42)
    linear.weight.data.normal_(0, 0.02)
    linear.bias.data.zero_()
    return linear

@pytest.fixture
def mock_model():
    """Create a simplified mock model structure for testing LoRA."""
    class MockAttention:
        def __init__(self):
            self.q_proj = torch.nn.Linear(32, 32)
            self.v_proj = torch.nn.Linear(32, 32)
    
    class MockLayer:
        def __init__(self):
            self.self_attn = MockAttention()
    
    class MockModel:
        def __init__(self):
            self.layers = [MockLayer(), MockLayer()]
            self.lm_head = torch.nn.Linear(32, 50000, bias=False)
            self.device = 'cpu'
            self.config = type('obj', (object,), {'vocab_size': 50000})
    
    model = MockModel()
    return model