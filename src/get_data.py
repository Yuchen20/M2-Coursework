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
    def __init__(self, data, processor, context_length=256, inference=False, target_eval_pairs = 10):
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

        # Preprocess all samples
        self.tokenized_data = []
        for i, sample in enumerate(self.data):
            assert sample.shape[1] == 2, f"Sample {i} should have shape [seq_length, 2], got {sample.shape}"
            # Process each time series sample to get tokenized representation
            tokenized = self.processor.preprocess(sample[:, 0], sample[:, 1])
            self.tokenized_data.append(tokenized)

        # For inference mode, prepare sliding windows in advance
        self.inference_data = []
        if inference:
            self._prepare_inference_data()
    
    def _prepare_inference_data(self):
        """
        Prepare data for inference by creating structured input-target pairs.
        
        For each sequence, this method:
        1. Identifies semicolon positions as breakpoints
        2. Creates sliding windows with context_length with stride of context_length/2
        3. For each window, finds appropriate input and target sequences bounded by semicolons
        4. Organizes these sequences into properly formatted samples for inference
        
        Raises:
            ValueError: If semicolon token cannot be determined or no valid sequences found
        """
        # Get semicolon token ID safely
        semicolon_token_id = self._get_semicolon_token_id()
        
        valid_sequences_count = 0
        max_target_length = 0
        
        # First pass to collect sample data and determine maximum target length
        temp_inference_data = []
        
        for sample_idx, sample in tqdm(enumerate(self.tokenized_data), desc="Processing sequences", leave=False):
            input_ids = sample['input_ids'].squeeze()
            
            # Find positions of all semicolon tokens
            semicolon_positions = (input_ids == semicolon_token_id).nonzero().squeeze(-1)
            
            # Ensure semicolon_positions has proper shape
            if semicolon_positions.numel() == 0:  # No semicolons found
                continue
            if len(semicolon_positions.shape) == 0:  # Single semicolon
                semicolon_positions = semicolon_positions.unsqueeze(0)
            
            # Need at least two semicolons to form input-target pair
            if len(semicolon_positions) < 2:
                continue
                
            valid_sequences_count += 1
            stride = 256

            # Process sliding windows
            for start_pos in range(0, len(input_ids) - self.context_length + 1, stride):
                # find the nearest semicolon after the start_pos
                relevant_semicolons = semicolon_positions[semicolon_positions >= start_pos]
                
                # No suitable semicolon after the window, skip this window
                if len(relevant_semicolons) == 0:
                    continue

                input_start_pos = relevant_semicolons[0] + 1
                input_end_pos   = input_start_pos + self.context_length

                # Find target end position (nearest semicolon after the window)
                target_semicolons = semicolon_positions[semicolon_positions >= input_end_pos]
                if len(target_semicolons) <= self.target_eval_pairs + 1:
                    continue

                input_sequence = input_ids[input_start_pos:input_end_pos]
                target_sequence = input_ids[input_end_pos:]
                
                # Skip if either sequence is empty
                if len(input_sequence) == 0 or len(target_sequence) == 0:
                    continue
                    
                # Track maximum target length
                if len(target_sequence) > max_target_length:
                    max_target_length = len(target_sequence)
                    
                temp_inference_data.append({
                    'input_ids': input_sequence,
                    'target': target_sequence,
                    'full_sequence': input_ids,
                })
        
        # Ensure maximum target length is reasonable to avoid excessive padding
        if max_target_length > 1024:  # Cap at a reasonable size
            max_target_length = 1024
        
        # Second pass to create properly padded data
        self.inference_data = []
        for item in temp_inference_data:
            input_sequence = item['input_ids']
            target_sequence = item['target']
            
            # Pad or truncate target to max_target_length
            if len(target_sequence) > max_target_length:
                target_sequence = target_sequence[:max_target_length]
            elif len(target_sequence) < max_target_length:
                padding = torch.full((max_target_length - len(target_sequence),), 
                                    self.processor.tokenizer.pad_token_id,
                                    dtype=target_sequence.dtype)
                target_sequence = torch.cat([target_sequence, padding])
            
            self.inference_data.append({
                'input_ids': input_sequence,
                'attention_mask': (input_sequence != self.processor.tokenizer.pad_token_id).float(),
                'target': target_sequence,
                'target_attention_mask': (target_sequence != self.processor.tokenizer.pad_token_id).float(),
            })
        # check again for the same length
        for data in self.inference_data:
            assert data['target'].shape[0] == max_target_length, f"Expected target length {max_target_length}, got {data['target'].shape[0]}"

        
        if valid_sequences_count == 0:
            raise ValueError("No sequences with semicolon tokens were found in the data.")
        elif len(self.inference_data) == 0:
            raise ValueError("Could not create any valid input-target pairs for inference.")
        
        print(f"Prepared {len(self.inference_data)} inference samples with max target length {max_target_length}")
                
    def _get_semicolon_token_id(self):
        """Helper method to safely get the semicolon token ID."""
        try:
            # First attempt: direct token conversion
            semicolon_token_id = self.processor.tokenizer.convert_tokens_to_ids([';'])[0]
        except (AttributeError, IndexError):
            try:
                # Second attempt: encoding
                semicolon_token_id = self.processor.tokenizer.encode(';', add_special_tokens=False)[0]
            except (AttributeError, IndexError):
                try:
                    # Third attempt: through vocabulary
                    semicolon_token_id = self.processor.tokenizer.vocab.get(';')
                    if semicolon_token_id is None:
                        raise ValueError()
                except (AttributeError, ValueError):
                    raise ValueError("Could not determine the token ID for ';'. Check tokenizer implementation.")
        return semicolon_token_id
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        if self.inference:
            return len(self.inference_data)
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to return
            
        Returns:
            dict: Dictionary containing input_ids, attention_mask, and target
        """
        if self.inference:
            return self.inference_data[idx]
        
        # Training mode: random window with causal next-token prediction
        tokenized_sample = self.tokenized_data[idx]
        input_ids = tokenized_sample['input_ids'].squeeze()
        
        if len(input_ids) <= self.context_length + 1:
            # If sequence is too short, use the whole sequence
            inputs = input_ids[:-1]
            targets = input_ids[1:]
        else:
            # Random sliding window for training
            max_start_idx = len(input_ids) - self.context_length - 1
            start_idx = np.random.randint(0, max_start_idx)
            end_idx = start_idx + self.context_length
            
            inputs = input_ids[start_idx:end_idx]
            targets = input_ids[start_idx+1:end_idx+1]

        inputs = inputs
             
        return {
            'input_ids': inputs,
            'attention_mask': (inputs != self.processor.tokenizer.pad_token_id).float(),
            'target': targets,
        }
        

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








