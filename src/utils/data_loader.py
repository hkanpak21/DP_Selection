import numpy as np
import pandas as pd
import os
from typing import Tuple, List, Dict, Optional, Union, Any

class DataLoader:
    """
    Utility for loading and preprocessing datasets for DP selection experiments.
    """
    
    def __init__(self, data_dir: str = "data/dpbench"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing datasets
        """
        self.data_dir = data_dir
        
    def list_available_datasets(self) -> List[str]:
        """
        List all available datasets in the data directory.
        
        Returns:
            List of dataset filenames
        """
        if not os.path.exists(self.data_dir):
            return []
            
        return [f for f in os.listdir(self.data_dir) 
                if f.endswith(('.csv', '.parquet', '.json'))]
    
    def load_csv_dataset(self, filename: str, 
                       target_column: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load a dataset from a CSV file.
        
        Args:
            filename: Name of the CSV file
            target_column: Name of the target column (if any)
            
        Returns:
            Tuple of (features, target) arrays
        """
        file_path = os.path.join(self.data_dir, filename)
        df = pd.read_csv(file_path)
        
        if target_column is not None and target_column in df.columns:
            y = df[target_column].values
            X = df.drop(columns=[target_column]).values
            return X, y
        else:
            return df.values, None
    
    def generate_synthetic_data(self, size: int = 100, 
                              distribution: str = "uniform", 
                              max_value: float = 1.0) -> np.ndarray:
        """
        Generate synthetic data for testing.
        
        Args:
            size: Size of the vector to generate
            distribution: Type of distribution ('uniform', 'normal', 'exponential')
            max_value: Maximum value in the generated data
            
        Returns:
            Generated data vector
        """
        if distribution == "uniform":
            data = np.random.uniform(0, max_value, size=size)
        elif distribution == "normal":
            data = np.random.normal(max_value/2, max_value/4, size=size)
            data = np.clip(data, 0, max_value)
        elif distribution == "exponential":
            data = np.random.exponential(scale=max_value/3, size=size)
            data = np.clip(data, 0, max_value)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
            
        return data
    
    def preprocess_for_dp_selection(self, data: np.ndarray, 
                                  normalization: str = "minmax") -> np.ndarray:
        """
        Preprocess data for DP selection experiments.
        
        Args:
            data: Input data array
            normalization: Type of normalization to apply
            
        Returns:
            Preprocessed data
        """
        if normalization == "minmax":
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val > min_val:
                return (data - min_val) / (max_val - min_val)
            else:
                return np.zeros_like(data)
        elif normalization == "l2":
            norm = np.linalg.norm(data)
            if norm > 0:
                return data / norm
            else:
                return data
        elif normalization is None or normalization == "none":
            return data
        else:
            raise ValueError(f"Unsupported normalization: {normalization}")
            
    def create_benchmark_datasets(self, sizes: List[int] = [10, 100, 1000], 
                              distributions: List[str] = ["uniform", "normal"],
                              num_samples: int = 5) -> Dict[str, np.ndarray]:
        """
        Create a collection of benchmark datasets.
        
        Args:
            sizes: List of vector sizes to generate
            distributions: List of distributions to use
            num_samples: Number of samples per configuration
            
        Returns:
            Dictionary of benchmark datasets
        """
        benchmarks = {}
        
        for size in sizes:
            for dist in distributions:
                for i in range(num_samples):
                    key = f"{dist}_{size}_{i}"
                    benchmarks[key] = self.generate_synthetic_data(size, dist)
                    
        return benchmarks 