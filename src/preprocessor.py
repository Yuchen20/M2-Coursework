from typing import List, Tuple, Union, Optional
import torch
import numpy as np
from transformers import PreTrainedTokenizer

class NumericalProcessor:
    """
    A processor for converting between numerical arrays and tokenized strings.
    Used for representing predator-prey data as text for language model processing.
    """
    
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer, 
                 data: np.ndarray, 
                 max_magnitude: float = 10.0, 
                 precision: int = 2) -> None:
        """
        Initialize the numerical processor.
        
        Args:
            tokenizer: The tokenizer to use for encoding and decoding text
            data: Training data used to calculate the scaling factor
            max_magnitude: Maximum value after scaling (to keep values in a reasonable range)
            precision: Number of decimal places to keep in string representation
        """
        self.tokenizer = tokenizer
        
        # Validate data
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array")
        if data.size == 0:
            raise ValueError("Data cannot be empty")
            
        # Handle potential NaN or inf values in the data
        clean_data = data.flatten()
        clean_data = clean_data[~np.isnan(clean_data) & ~np.isinf(clean_data)]
        
        if clean_data.size == 0:
            raise ValueError("After filtering invalid values, no data remains")
            
        # Keep 3 sigma data within max_magnitude
        self.scaler = max_magnitude / np.percentile(clean_data, 99.7)
        self.precision = precision
        
        # Validate precision
        if not isinstance(precision, int) or precision < 0:
            raise ValueError("Precision must be a non-negative integer")
            
    def _to_string(self, prey: np.ndarray, predator: np.ndarray) -> str:
        """
        Convert prey and predator arrays to a formatted string representation.
        
        Args:
            prey: Array of prey population values
            predator: Array of predator population values
            
        Returns:
            String representation of the data in format "p1,p2;p1,p2;..."
        
        Raises:
            ValueError: If arrays have different lengths or contain invalid values
        """
        if len(prey) != len(predator):
            raise ValueError(f"Prey and predator arrays must have the same length, got {len(prey)} and {len(predator)}")
            
        if len(prey) == 0:
            return ""
            
        # Handle NaN and inf values
        prey = np.nan_to_num(prey * self.scaler, nan=0.0, posinf=self.scaler, neginf=-self.scaler)
        predator = np.nan_to_num(predator * self.scaler, nan=0.0, posinf=self.scaler, neginf=-self.scaler)
        
        formatted_pairs = [f'{p1:.{self.precision}f},{p2:.{self.precision}f}' 
                          for p1, p2 in zip(prey, predator)]
        return ';'.join(formatted_pairs) + ';'
        
    def preprocess(self, prey: np.ndarray, predator: np.ndarray) -> dict:
        """
        Convert prey and predator data to tokenized input for the model.
        
        Args:
            prey: Array of prey population values
            predator: Array of predator population values
            
        Returns:
            Dictionary with tokenized inputs ready for model consumption
        """
        text = self._to_string(prey, predator)
        return self.tokenizer(text, return_tensors='pt')
    
    def postprocess(self, output: Union[List[int], torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Convert model output tokens back to numerical values.
        
        Args:
            output: Token IDs from the model
            
        Returns:
            Array of shape (N, 2) containing the descaled prey and predator values
            
        Raises:
            ValueError: If the output format is invalid
        """
        # Convert to list if tensor or ndarray
        if isinstance(output, (torch.Tensor, np.ndarray)):
            output = output.tolist()
            
        # Decode token IDs to text
        try:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
        except Exception as e:
            raise ValueError(f"Failed to decode tokens: {e}")
            
        # Parse the text format
        if not text or ';' not in text:
            return np.zeros((0, 2), dtype=np.float32)
            
        # Process each value pair
        try:
            # Split by semicolon and remove trailing empty element
            pairs = text.split(';')
            if pairs[-1] == '':
                pairs = pairs[:-1]
                
            # Parse each pair into prey,predator values
            values = []
            for pair in pairs:
                if ',' in pair:
                    p1, p2 = pair.split(',', 1)
                    values.append([float(p1), float(p2)])
                    
            # Convert to numpy array and descale
            if values:
                values_array = np.array(values, dtype=np.float32)
                return values_array / self.scaler
            else:
                return np.zeros((0, 2), dtype=np.float32)
                
        except Exception as e:
            raise ValueError(f"Failed to parse output text '{text[:50]}...': {e}")

    def decode_to_string(self, ids: Union[List[int], torch.Tensor]) -> str:
        """
        Decode token IDs to string representation (without parsing to values).
        
        Args:
            ids: Token IDs to decode
            
        Returns:
            Raw decoded string
        """
        return self.tokenizer.decode(ids, skip_special_tokens=True)