import numpy as np
from typing import Dict, Any, List, Tuple

def compute_error_metrics(true_values: np.ndarray, selected_indices: List[int]) -> Dict[str, Any]:
    """
    Compute error metrics between true maximum and selected values.
    
    Args:
        true_values: Original data vector
        selected_indices: List of selected indices from DP mechanisms
        
    Returns:
        Dictionary with error metrics including average error, accuracy, etc.
    """
    if len(selected_indices) == 0:
        return {
            "average_error": float('inf'),
            "accuracy": 0.0,
            "std_error": float('inf'),
        }
    
    # Find true maximum
    true_max_index = np.argmax(true_values)
    true_max_value = true_values[true_max_index]
    
    # Compute errors
    errors = [abs(true_max_value - true_values[idx]) for idx in selected_indices]
    
    # Compute accuracy (percentage of times we select the true max)
    accuracy = selected_indices.count(true_max_index) / len(selected_indices)
    
    return {
        "average_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "std_error": float(np.std(errors)),
        "max_error": float(np.max(errors)),
        "min_error": float(np.min(errors)),
        "accuracy": float(accuracy),
    } 