import pytest
import os
import numpy as np
import pandas as pd
from get_flops import QwenFlopsCalculator, OperationFlops

class TestQwenFlopsCalculator:
    """Tests for the QwenFlopsCalculator class."""
    
    def test_initialization(self):
        """Test initialization of the QwenFlopsCalculator."""
        calculator = QwenFlopsCalculator()
        
        # Check that calculator has expected attributes
        assert hasattr(calculator, 'num_layers')
        assert hasattr(calculator, 'hidden_dim')
        assert hasattr(calculator, 'num_heads')
        assert hasattr(calculator, 'head_dim')
        assert hasattr(calculator, 'ffn_dim')
        assert hasattr(calculator, 'vocab_size')
        assert hasattr(calculator, 'opFLops')
        assert hasattr(calculator, 'log_file')
        
        # Check log file was created
        assert os.path.exists(calculator.log_folder)
        assert os.path.exists(calculator.log_file)
    
    def test_attention_flops(self):
        """Test calculation of attention FLOPs."""
        calculator = QwenFlopsCalculator()
        
        # Calculate attention FLOPs
        batch_size = 2
        seq_len = 16
        flops, breakdown = calculator.get_attention_flops(batch_size, seq_len)
        
        # Check that FLOPs is positive
        assert flops > 0
        
        # Check that breakdown is a non-empty string
        assert isinstance(breakdown, str)
        assert len(breakdown) > 0
        
        # Check that different inputs produce different FLOPs counts
        flops2, _ = calculator.get_attention_flops(batch_size * 2, seq_len)
        flops3, _ = calculator.get_attention_flops(batch_size, seq_len * 2)
        
        # More inputs should result in more FLOPs
        assert flops2 > flops
        assert flops3 > flops
    
    def test_rmsnorm_flops(self):
        """Test calculation of RMSNorm FLOPs."""
        calculator = QwenFlopsCalculator()
        
        # Calculate RMSNorm FLOPs
        batch_size = 2
        seq_len = 16
        flops, breakdown = calculator.get_RMSNorm_flops(batch_size, seq_len)
        
        # Check that FLOPs is positive
        assert flops > 0
        
        # Check that breakdown is a non-empty string
        assert isinstance(breakdown, str)
        assert len(breakdown) > 0
        
        # Check that different inputs produce different FLOPs counts
        flops2, _ = calculator.get_RMSNorm_flops(batch_size * 2, seq_len)
        
        # More inputs should result in more FLOPs
        assert flops2 > flops
    
    def test_ffn_flops(self):
        """Test calculation of FFN FLOPs."""
        calculator = QwenFlopsCalculator()
        
        # Calculate FFN FLOPs
        batch_size = 2
        seq_len = 16
        flops, breakdown = calculator.get_ffn_flops(batch_size, seq_len)
        
        # Check that FLOPs is positive
        assert flops > 0
        
        # Check that breakdown is a non-empty string
        assert isinstance(breakdown, str)
        assert len(breakdown) > 0
        
        # Check that different inputs produce different FLOPs counts
        flops2, _ = calculator.get_ffn_flops(batch_size * 2, seq_len)
        
        # More inputs should result in more FLOPs
        assert flops2 > flops
    
    def test_lora_flops(self):
        """Test calculation of LoRA FLOPs."""
        calculator = QwenFlopsCalculator()
        
        # Calculate LoRA FLOPs
        batch_size = 2
        seq_len = 16
        rank = 4
        flops, breakdown = calculator.get_lora_flops(batch_size, seq_len, rank)
        
        # Check that FLOPs is positive
        assert flops > 0
        
        # Check that breakdown is a non-empty string
        assert isinstance(breakdown, str)
        assert len(breakdown) > 0
        
        # Check that different ranks produce different FLOPs counts
        flops2, _ = calculator.get_lora_flops(batch_size, seq_len, rank * 2)
        
        # Higher rank should result in more FLOPs
        assert flops2 > flops
    
    def test_get_flops(self):
        """Test the main get_flops method."""
        calculator = QwenFlopsCalculator()
        
        # Calculate total FLOPs
        batch_size = 2
        seq_len = 16
        rank = 4
        flops = calculator.get_flops(batch_size, seq_len, rank)
        
        # Check that FLOPs is positive
        assert flops > 0
        
        # Check that different inputs produce different FLOPs counts
        flops2 = calculator.get_flops(batch_size * 2, seq_len, rank)
        flops3 = calculator.get_flops(batch_size, seq_len, rank * 2)
        
        # More inputs should result in more FLOPs
        assert flops2 > flops
        assert flops3 > flops
        
        # Test inference mode (should have fewer FLOPs than training)
        flops_inference = calculator.get_flops(batch_size, seq_len, rank, inference=True)
        assert flops_inference < flops
    
    def test_log_flops(self):
        """Test logging FLOPs to a file."""
        calculator = QwenFlopsCalculator()
        
        # Initial file state
        initial_df = pd.read_csv(calculator.log_file)
        initial_row_count = len(initial_df)
        
        # Log some FLOPs
        calculator.log_flops(batch_size=2, seq_len=16, rank=4, description="test")
        
        # Check that a new row was added to the log file
        updated_df = pd.read_csv(calculator.log_file)
        assert len(updated_df) > initial_row_count
        
        # Check that the latest entry has the correct log name
        latest_entry = updated_df[updated_df['name'] == calculator.log_name]
        assert not latest_entry.empty
        assert latest_entry.iloc[-1]['description'] == "test"

class TestOperationFlops:
    """Tests for the OperationFlops dataclass."""
    
    def test_operation_flops_values(self):
        """Test that OperationFlops has the expected FLOP values."""
        op_flops = OperationFlops()
        
        # Test basic operations
        assert op_flops._ADD == 1
        assert op_flops._SUB == 1
        assert op_flops._NEG == 1
        assert op_flops._MUL == 1
        assert op_flops._DIV == 1
        assert op_flops._INV == 1
        assert op_flops._RELU == 1
        assert op_flops._ABS == 1
        
        # Test more complex operations
        assert op_flops._EXP == 10
        assert op_flops._LOG == 10
        assert op_flops._SIN == 10
        assert op_flops._COS == 10
        assert op_flops._SQRT == 10