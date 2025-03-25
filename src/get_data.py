import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessor import NumericalProcessor
from tqdm import tqdm
import torch

# set random seed for reproducibility
np.random.seed(42)



class LotkaVolterraDataset(Dataset):
    """Dataset class for Lotka-Volterra time series data.

    This class handles preprocessing and serving time series data for training and inference
    of models on Lotka-Volterra dynamics.

    Args:
        data (List[np.ndarray]): List of time series arrays, each with shape [sequence_length, 2]
        processor: Preprocessing object with a preprocess method
        context_length (int, optional): Length of context window. Defaults to 256.
        inference (bool, optional): Whether to prepare data for inference. Defaults to False.

    Raises:
        AssertionError: If inputs don't meet requirements
    """
    def __init__(self, data, processor, context_length=256, inference=False, target_eval_pairs=10):
        super().__init__()
        # Input validation
        assert data is not None and len(data) > 0, "Data must be non-empty"
        assert hasattr(processor, 'preprocess'), "Processor must have preprocess method"
        assert context_length > 0, "Context length must be positive"
        
        self.data = data
        self.processor = processor
        self.context_length = context_length
        self.inference = inference
        self.target_eval_pairs = target_eval_pairs
        self.stride = 256

        # Preprocess all samples and chunk with fixed stride
        self.processed_chunks = []
        
        for i, sample in enumerate(tqdm(self.data, desc="Processing data", leave=False)):
            assert sample.shape[1] == 2, f"Sample {i} should have shape [seq_length, 2], got {sample.shape}"
            # Process each time series sample to get tokenized representation
            tokenized = self.processor.preprocess(sample[:, 0], sample[:, 1])
            input_ids = tokenized['input_ids'].squeeze()
            
            # Skip if sequence is too short
            if len(input_ids) <= self.context_length:
                continue
                
            # Create chunks with fixed stride
            for start_idx in range(0, len(input_ids) - self.context_length, self.stride):
                end_idx = start_idx + self.context_length
                
                # For training, we need the target to be the next tokens
                if not inference:
                    if end_idx + 1 > len(input_ids):
                        # Skip if we can't get the next token for the last position
                        continue
                    
                    input_chunk = input_ids[start_idx:end_idx]
                    target_chunk = input_ids[start_idx+1:end_idx+1]
                    
                    self.processed_chunks.append({
                        'input_ids': input_chunk,
                        'attention_mask': (input_chunk != self.processor.tokenizer.pad_token_id).float(),
                        'target': target_chunk,
                    })
                # For inference, we need to include more future tokens for evaluation
                else:
                    input_chunk = input_ids[start_idx:end_idx]
                    
                    # Make sure we have enough future tokens for evaluation
                    target_length = min(self.context_length, len(input_ids) - end_idx)
                    if target_length <= 0:
                        continue
                        
                    target_chunk = input_ids[end_idx:end_idx+target_length]
                    
                    self.processed_chunks.append({
                        'input_ids': input_chunk,
                        'attention_mask': (input_chunk != self.processor.tokenizer.pad_token_id).float(),
                        'target': target_chunk,
                        'full_sequence': []
                    })
        
        print(f"Created {len(self.processed_chunks)} chunks from {len(self.data)} sequences")
        if len(self.processed_chunks) == 0:
            raise ValueError("No valid chunks were created. Check your data and context length.")
    
    def __len__(self):
        """Return the number of chunks in the dataset."""
        return len(self.processed_chunks)
    
    def __getitem__(self, idx):
        """Get a chunk from the dataset.
        
        Args:
            idx (int): Index of the chunk to return
            
        Returns:
            dict: Dictionary containing input_ids, attention_mask, and target
        """
        return self.processed_chunks[idx]
        

class DataMaster:
    def __init__(self, tokenizer, trajectories, processor=None, test_size=0.2, val_size=0.1, experiment_fraction=0.1):
        self.tokenizer = tokenizer
        self.trajectories = trajectories
        self.processor = processor
        self.test_size = test_size
        self.val_size = val_size
        self.experiment_fraction = experiment_fraction

        self.train_trajectories, self.test_trajectories = train_test_split(trajectories, test_size=test_size + val_size, random_state=42)
        self.val_trajectories, self.test_trajectories = train_test_split(self.test_trajectories, test_size=val_size / (test_size + val_size), random_state=42)

        self.processor = NumericalProcessor(tokenizer, self.train_trajectories)

        if experiment_fraction == 1.0:
            self.exp_train_trajectories = self.train_trajectories
            self.exp_valid_trajectories = self.val_trajectories
            self.exp_test_trajectories = self.test_trajectories
            return

        # experiment dataset
        self.exp_train_trajectories, _ = train_test_split(self.train_trajectories, test_size=1 - experiment_fraction, random_state=42)
        self.exp_valid_trajectories, _ = train_test_split(self.val_trajectories, test_size=1 - experiment_fraction, random_state=42)
        self.exp_test_trajectories, _ = train_test_split(self.test_trajectories, test_size=1 - experiment_fraction, random_state=42)


    def get_data(self, experiment=False, context_length=128, batch_size=4, target_eval_pairs=10):
        
        if experiment:
            train_trajectories = self.exp_train_trajectories
            val_trajectories = self.exp_valid_trajectories
            test_trajectories = self.exp_test_trajectories
        else:
            train_trajectories = self.train_trajectories
            val_trajectories = self.val_trajectories
            test_trajectories = self.test_trajectories

        train_dataset = LotkaVolterraDataset(train_trajectories, self.processor, context_length=context_length, inference=False, target_eval_pairs=target_eval_pairs)
        val_dataset = LotkaVolterraDataset(val_trajectories, self.processor, context_length=context_length, inference=True, target_eval_pairs=target_eval_pairs)    
        test_dataset = LotkaVolterraDataset(test_trajectories, self.processor, context_length=context_length, inference=True, target_eval_pairs=target_eval_pairs)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # validate the data
        for i, loader in enumerate(
            tqdm(
            [train_loader, val_loader, test_loader],
            desc="Sanity check",
            leave=False
        )
        ):
            for data in tqdm(
                loader, 
                desc=f'Sanity check {["Train", "Val", "Test"][i]}',
                leave=True
            ):
                assert data['input_ids'].shape[1] == context_length, f"Expected context length {context_length}, got {data['input_ids'].shape[1]}"

        return train_loader, val_loader, test_loader








