import pytest
import numpy as np
import torch
from get_data import LotkaVolterraDataset, DataMaster
from preprocessor import NumericalProcessor

class TestLotkaVolterraDataset:
    """Tests for the LotkaVolterraDataset class."""
    
    def test_initialization(self, sample_data, tokenizer):
        """Test dataset initialization with various parameters."""
        processor = NumericalProcessor(tokenizer, sample_data)
        
        # Test with default parameters
        dataset = LotkaVolterraDataset(sample_data, processor)
        assert len(dataset) > 0
        
        # Test with custom context length
        context_length = 64
        dataset = LotkaVolterraDataset(sample_data, processor, context_length=context_length)
        assert len(dataset) > 0
        
        # Check that context length is respected
        sample = dataset[0]
        assert sample['input_ids'].shape[0] == context_length
    
    def test_getitem(self, sample_data, tokenizer):
        """Test the __getitem__ method of the dataset."""
        processor = NumericalProcessor(tokenizer, sample_data)
        context_length = 64
        dataset = LotkaVolterraDataset(sample_data, processor, context_length=context_length)
        
        # Get a sample
        sample = dataset[0]
        
        # Check that it contains expected keys
        assert 'input_ids' in sample
        assert 'target' in sample
        
        # Check shapes
        assert sample['input_ids'].shape[0] == context_length
        assert sample['target'].shape[0] == context_length
    
    def test_inference_mode(self, sample_data, tokenizer):
        """Test dataset in inference mode."""
        processor = NumericalProcessor(tokenizer, sample_data)
        context_length = 64
        target_eval_pairs = 5
        
        # Create dataset in inference mode
        dataset = LotkaVolterraDataset(
            sample_data, 
            processor, 
            context_length=context_length,
            inference=True,
            target_eval_pairs=target_eval_pairs
        )
        
        # Get a sample
        sample = dataset[0]
        
        # Check that it contains expected keys
        assert 'input_ids' in sample
        assert 'target' in sample
        
        # Check shapes - target should be longer in inference mode
        assert sample['input_ids'].shape[0] == context_length
        assert sample['target'].shape[0] > context_length  # Target includes future tokens
    
    def test_error_handling(self, tokenizer):
        """Test error handling for invalid inputs."""
        processor = NumericalProcessor(tokenizer, np.ones((1, 10, 2)))
        
        # Test with empty data
        with pytest.raises(AssertionError):
            LotkaVolterraDataset([], processor)
        
        # Test with invalid processor (no preprocess method)
        class InvalidProcessor:
            pass
            
        with pytest.raises(AssertionError):
            LotkaVolterraDataset(np.ones((1, 10, 2)), InvalidProcessor())
        
        # Test with invalid context length
        with pytest.raises(AssertionError):
            LotkaVolterraDataset(np.ones((1, 10, 2)), processor, context_length=0)


class TestDataMaster:
    """Tests for the DataMaster class."""
    
    def test_initialization(self, sample_data, tokenizer):
        """Test DataMaster initialization with various parameters."""
        # Test with default parameters
        data_master = DataMaster(tokenizer, sample_data)
        
        # Check attributes
        assert hasattr(data_master, 'tokenizer')
        assert hasattr(data_master, 'trajectories')
        assert hasattr(data_master, 'processor')
        assert hasattr(data_master, 'train_trajectories')
        assert hasattr(data_master, 'val_trajectories')
        assert hasattr(data_master, 'test_trajectories')
        
        # Check data splits
        assert len(data_master.train_trajectories) > 0
        assert len(data_master.val_trajectories) > 0
        assert len(data_master.test_trajectories) > 0
        
        # Check with experiment_fraction < 1.0
        data_master = DataMaster(tokenizer, sample_data, experiment_fraction=0.5)
        assert len(data_master.exp_train_trajectories) < len(data_master.train_trajectories)
    
    def test_get_data(self, sample_data, tokenizer):
        """Test getting data loaders from DataMaster."""
        data_master = DataMaster(tokenizer, sample_data)
        
        # Get data loaders
        train_loader, val_loader, test_loader = data_master.get_data(
            context_length=32,
            batch_size=2,
            experiment=False
        )
        
        # Check that loaders are valid
        assert hasattr(train_loader, '__iter__')
        assert hasattr(val_loader, '__iter__')
        assert hasattr(test_loader, '__iter__')
        
        # Check experimental mode
        exp_train_loader, exp_val_loader, exp_test_loader = data_master.get_data(
            context_length=32,
            batch_size=2,
            experiment=True
        )
        
        # Check that loaders are valid
        assert hasattr(exp_train_loader, '__iter__')
        assert hasattr(exp_val_loader, '__iter__')
        assert hasattr(exp_test_loader, '__iter__')
    
    def test_batch_properties(self, sample_data, tokenizer):
        """Test properties of batches produced by DataMaster."""
        data_master = DataMaster(tokenizer, sample_data)
        context_length = 32
        batch_size = 2
        
        # Get data loaders
        train_loader, _, _ = data_master.get_data(
            context_length=context_length,
            batch_size=batch_size,
            experiment=False
        )
        
        # Check first batch
        try:
            batch = next(iter(train_loader))
            
            # Check batch size
            assert batch['input_ids'].shape[0] == batch_size
            
            # Check context length
            assert batch['input_ids'].shape[1] == context_length
            
            # Check that target has the same length as input_ids for training
            assert batch['target'].shape == batch['input_ids'].shape
            
        except StopIteration:
            # If the loader is empty, skip this test
            pytest.skip("Data loader is empty, cannot test batch properties")