import numpy as np
from typing import Union, Tuple, List, Optional
import math

def sample_geometric(p: float, size: int = 1) -> Union[int, np.ndarray]:
    """
    Sample from a one-sided geometric distribution.
    
    Args:
        p: Success probability parameter
        size: Number of samples
        
    Returns:
        Sampled value(s)
    """
    return np.random.geometric(p, size=size) - 1

def sample_negative_binomial(r: int, p: float, size: int = 1) -> Union[int, np.ndarray]:
    """
    Sample from a negative binomial distribution.
    
    Args:
        r: Number of successes
        p: Success probability parameter
        size: Number of samples
        
    Returns:
        Sampled value(s)
    """
    return np.random.negative_binomial(r, p, size=size)

def round_to_bits(values: np.ndarray, bits: int) -> np.ndarray:
    """
    Round values to a specified number of bits of precision.
    
    Args:
        values: Array of values to round
        bits: Number of bits of precision
        
    Returns:
        Rounded values
    """
    scale = 2 ** bits
    return np.floor(values * scale) / scale

def noise_calibration(epsilon: float, sensitivity: float = 1.0, 
                     mechanism: str = "geometric") -> float:
    """
    Calibrate noise parameter based on privacy parameters.
    
    Args:
        epsilon: Privacy parameter (ε)
        sensitivity: Sensitivity of the function
        mechanism: Type of noise mechanism
        
    Returns:
        Calibrated parameter for the noise distribution
    """
    if mechanism == "geometric":
        return 1 - np.exp(-epsilon / sensitivity)
    elif mechanism == "laplace":
        return sensitivity / epsilon
    else:
        raise ValueError(f"Unsupported mechanism: {mechanism}")

def compute_error_metrics(true_values: np.ndarray, estimated_values: np.ndarray, 
                        true_max_index: int, estimated_max_index: int) -> dict:
    """
    Compute error metrics for DP selection.
    
    Args:
        true_values: Ground truth values
        estimated_values: DP-noised values
        true_max_index: Ground truth argmax
        estimated_max_index: DP argmax
        
    Returns:
        Dictionary of error metrics
    """
    metrics = {
        "correct_selection": true_max_index == estimated_max_index,
        "max_error": abs(true_values[true_max_index] - estimated_values[estimated_max_index]),
        "l1_error": np.mean(np.abs(true_values - estimated_values)),
        "l2_error": np.sqrt(np.mean(np.square(true_values - estimated_values))),
        "max_relative_error": abs(true_values[true_max_index] - estimated_values[estimated_max_index]) / abs(true_values[true_max_index]) if true_values[true_max_index] != 0 else float('inf')
    }
    
    return metrics 