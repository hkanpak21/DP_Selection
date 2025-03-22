import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from .noise_and_round import NoiseAndRound
from .dp_mechanisms import (
    ExponentialMechanism, 
    PermuteAndFlip,
    RandomizedResponse,
    RandomChoice,
    NoiseAndRoundWithoutRounding,
    SecureAggregation
)

class DPSelection:
    """
    Wrapper module for differentially private selection mechanisms in the central setting.
    """
    
    def __init__(self, mechanism: str = "noise_and_round", **kwargs):
        """
        Initialize a DP selection mechanism.
        
        Args:
            mechanism: Type of mechanism to use
            **kwargs: Additional parameters for the specific mechanism
                - epsilon: Privacy parameter (required for all mechanisms)
                - sensitivity: Sensitivity parameter (for exponential mechanism)
                - num_parties: Number of parties (for secure aggregation)
                - r: Rounding bits (for noise_and_round variants)
        """
        self.mechanism_type = mechanism
        self.params = kwargs
        
        if "epsilon" not in kwargs and mechanism != "random_choice":
            raise ValueError("Epsilon parameter required for most mechanisms")
        
        if mechanism == "noise_and_round":
            self.mechanism = NoiseAndRound(**kwargs)
        elif mechanism == "noise_and_round_no_rounding":
            self.mechanism = NoiseAndRoundWithoutRounding(**kwargs)
        elif mechanism == "exponential":
            self.mechanism = ExponentialMechanism(**kwargs)
        elif mechanism == "permute_and_flip":
            self.mechanism = PermuteAndFlip(**kwargs)
        elif mechanism == "randomized_response":
            self.mechanism = RandomizedResponse(**kwargs)
        elif mechanism == "random_choice":
            self.mechanism = RandomChoice(**kwargs)
        elif mechanism == "secure_aggregation":
            self.mechanism = SecureAggregation(**kwargs)
        else:
            raise ValueError(f"Unsupported mechanism: {mechanism}")
    
    def select(self, data: np.ndarray, **kwargs) -> Tuple[int, float]:
        """
        Select an index from the data with differential privacy guarantees.
        
        Args:
            data: Input data vector
            **kwargs: Additional parameters for selection
            
        Returns:
            Tuple of (selected_index, score)
        """
        return self.mechanism.select(data, **kwargs)
    
    def evaluate_error(self, data: np.ndarray, true_max_index: int, 
                      num_trials: int = 100) -> Dict[str, Any]:
        """
        Evaluate error properties of the mechanism on the given data.
        
        Args:
            data: Input data vector
            true_max_index: Ground truth maximum index
            num_trials: Number of trials for error evaluation
            
        Returns:
            Dictionary with error statistics
        """
        results = {
            "accuracy": 0,
            "average_error": 0,
            "trials": []
        }
        
        true_max_value = data[true_max_index]
        
        for _ in range(num_trials):
            selected_idx, selected_value = self.select(data)
            error = abs(true_max_value - data[selected_idx])
            
            results["trials"].append({
                "selected_index": selected_idx,
                "error": error,
                "correct": selected_idx == true_max_index
            })
            
            if selected_idx == true_max_index:
                results["accuracy"] += 1
            
            results["average_error"] += error
        
        results["accuracy"] /= num_trials
        results["average_error"] /= num_trials
        
        return results 