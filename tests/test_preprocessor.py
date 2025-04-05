import pytest
import numpy as np
import torch
from preprocessor import NumericalProcessor

class TestNumericalProcessor:

    def test_initialization(self, sample_data, tokenizer):
        """Test that the NumericalProcessor initializes correctly."""
        processor = NumericalProcessor(tokenizer, sample_data)
        assert hasattr(processor, 'tokenizer')
        assert hasattr(processor, 'scaler')
        assert hasattr(processor, 'precision')
        assert processor.precision == 2  # Default precision

    def test_to_string(self, sample_data, tokenizer):
        """Test the _to_string method for formatting pairs of values."""
        processor = NumericalProcessor(tokenizer, sample_data)
        
        # Test simple case
        prey = np.array([1.0, 2.0, 3.0])
        predator = np.array([0.5, 1.5, 2.5])
        result = processor._to_string(prey, predator)
        
        # Verify format and scaling
        values = result.split(';')[:-1]  # Remove trailing empty item
        for i, value_pair in enumerate(values):
            p1, p2 = value_pair.split(',')
            # Check if scaled values match expected format (with precision 2)
            np.testing.assert_almost_equal(float(p1), prey[i] * processor.scaler, decimal=2)
            np.testing.assert_almost_equal(float(p2), predator[i] * processor.scaler, decimal=2)

    def test_preprocess(self, tokenizer):
        """Test preprocessing prey and predator data to tokenized form."""
        # Create simple test data
        data = np.array([[np.array([1.0, 2.0, 3.0]), np.array([0.5, 1.5, 2.5])]])
        processor = NumericalProcessor(tokenizer, data)
        
        # Test preprocessing
        prey = np.array([1.0, 2.0])
        predator = np.array([0.5, 1.5])
        result = processor.preprocess(prey, predator)
        
        # Check that result contains expected keys
        assert 'input_ids' in result
        # Verify tensor dimensions
        assert isinstance(result['input_ids'], torch.Tensor)

    def test_postprocess(self, sample_data, tokenizer):
        """Test postprocessing tokenized output back to numerical values."""
        processor = NumericalProcessor(tokenizer, sample_data)
        
        # Create test sequence
        prey = np.array([1.0, 2.0, 3.0])
        predator = np.array([0.5, 1.5, 2.5])
        
        # Convert to string and tokenize
        text = processor._to_string(prey, predator)
        tokens = tokenizer.encode(text)
        
        # Test postprocess
        result = processor.postprocess(tokens)
        
        # Check shape and values
        assert result.shape == (3, 2)
        np.testing.assert_almost_equal(result[:, 0], prey, decimal=1)
        np.testing.assert_almost_equal(result[:, 1], predator, decimal=1)

    def test_decode_to_string(self, sample_data, tokenizer):
        """Test decoding token IDs directly to string without parsing to values."""
        processor = NumericalProcessor(tokenizer, sample_data)
        
        # Create test sequence
        prey = np.array([1.0, 2.0])
        predator = np.array([0.5, 1.5])
        
        # Convert to string and tokenize
        text = processor._to_string(prey, predator)
        tokens = tokenizer.encode(text)
        
        # Test decode_to_string
        result = processor.decode_to_string(tokens)
        
        # Check that result is the expected string format
        assert ';' in result
        assert ',' in result

    def test_error_handling(self, tokenizer):
        """Test error handling for edge cases."""
        # Test with empty data
        with pytest.raises(ValueError):
            NumericalProcessor(tokenizer, np.array([]))
        
        # Test with valid processor but mismatched arrays
        processor = NumericalProcessor(tokenizer, np.ones((1, 100, 2)))
        with pytest.raises(ValueError):
            processor._to_string(np.array([1.0, 2.0]), np.array([0.5]))