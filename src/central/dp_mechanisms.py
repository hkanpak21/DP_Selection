import numpy as np
from typing import Tuple, List, Optional, Union
from .noise_and_round import NoiseAndRound

class ExponentialMechanism:
    """
    Implements the Exponential Mechanism for differentially private selection.
    """
    
    def __init__(self, epsilon: float, sensitivity: float = 1.0):
        """
        Initialize the Exponential Mechanism.
        
        Args:
            epsilon: Privacy parameter (ε)
            sensitivity: Sensitivity of the utility function
        """
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        
    def select(self, values: np.ndarray, **kwargs) -> Tuple[int, float]:
        """
        Select an index from the data with differential privacy guarantees.
        
        Args:
            values: Input data vector
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (selected_index, score)
        """
        n = len(values)
        # The utility function is the value itself
        utility = values
        
        # Using log-space calculations to avoid overflow
        scores = self.epsilon * utility / (2 * self.sensitivity)
        
        # Shift scores to avoid underflow in exp (subtract max score)
        scores_shifted = scores - np.max(scores)
        
        # Compute probabilities in a numerically stable way
        weights = np.exp(scores_shifted)
        probabilities = weights / np.sum(weights)
        
        # Check for NaN values and handle them
        if np.any(np.isnan(probabilities)):
            # Fallback: use a uniform distribution
            probabilities = np.ones(n) / n
            
        # Sample from the distribution
        selected_index = np.random.choice(n, p=probabilities)
        
        return int(selected_index), float(values[selected_index])

class PermuteAndFlip:
    """
    Implements the Permute-and-flip Mechanism for differentially private selection.
    """
    
    def __init__(self, epsilon: float):
        """
        Initialize the Permute-and-flip Mechanism.
        
        Args:
            epsilon: Privacy parameter (ε)
        """
        self.epsilon = epsilon
        
    def select(self, values: np.ndarray, **kwargs) -> Tuple[int, float]:
        """
        Select an index from the data with differential privacy guarantees.
        
        Args:
            values: Input data vector
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (selected_index, score)
        """
        n = len(values)
        indices = np.arange(n)
        
        # Generate random values for the permutation
        permutation_values = np.random.random(n)
        
        # Sort indices based on (values - random * 2/epsilon)
        sorted_indices = indices[np.argsort(values - permutation_values * (2.0 / self.epsilon))[::-1]]
        
        # The first index in the sorted list is our selection
        selected_index = sorted_indices[0]
        
        return int(selected_index), float(values[selected_index])

class RandomizedResponse:
    """
    Implements a Randomized Response Mechanism for differentially private selection.
    """
    
    def __init__(self, epsilon: float):
        """
        Initialize the Randomized Response Mechanism.
        
        Args:
            epsilon: Privacy parameter (ε)
        """
        self.epsilon = epsilon
        
    def select(self, values: np.ndarray, **kwargs) -> Tuple[int, float]:
        """
        Select an index from the data with differential privacy guarantees.
        
        Args:
            values: Input data vector
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (selected_index, score)
        """
        n = len(values)
        # Find the true max
        true_max_index = np.argmax(values)
        
        # Probability of reporting the true max
        p_true = np.exp(self.epsilon) / (np.exp(self.epsilon) + n - 1)
        
        # With probability p_true, return the true max, otherwise return a random index
        if np.random.random() < p_true:
            selected_index = true_max_index
        else:
            # Select a random index other than the true max
            other_indices = np.delete(np.arange(n), true_max_index)
            selected_index = np.random.choice(other_indices)
        
        return int(selected_index), float(values[selected_index])

class RandomChoice:
    """
    Implements a simple Random Choice Mechanism (baseline, non-private).
    """
    
    def __init__(self, epsilon: float = None):
        """
        Initialize the Random Choice Mechanism.
        
        Args:
            epsilon: Not used, included for compatibility
        """
        self.epsilon = epsilon
        
    def select(self, values: np.ndarray, **kwargs) -> Tuple[int, float]:
        """
        Select a random index from the data.
        
        Args:
            values: Input data vector
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (selected_index, score)
        """
        n = len(values)
        selected_index = np.random.randint(0, n)
        
        return int(selected_index), float(values[selected_index])

class NoiseAndRoundWithoutRounding:
    """
    Implements the Noise-and-round algorithm without the rounding step.
    """
    
    def __init__(self, epsilon: float, delta: Optional[float] = None, bits: int = 8):
        """
        Initialize the mechanism.
        
        Args:
            epsilon: Privacy parameter (ε)
            delta: Optional relaxation parameter for approximate DP
            bits: Number of bits for rounding (not used, included for API compatibility)
        """
        self.epsilon = epsilon
        self.delta = delta
        self.bits = bits  # Not used, but kept for API compatibility
        
    def _sample_geometric_noise(self, size: int = 1) -> Union[float, np.ndarray]:
        """
        Sample noise from a one-sided geometric distribution.
        
        Args:
            size: Number of noise samples to generate
            
        Returns:
            Sampled noise value(s)
        """
        p = 1 - np.exp(-self.epsilon)
        return np.random.geometric(p, size=size) - 1
        
    def select(self, values: np.ndarray, **kwargs) -> Tuple[int, float]:
        """
        Select the approximate maximum entry in a vector with DP guarantees.
        
        Args:
            values: Input vector
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (selected_index, noisy_value)
        """
        n = len(values)
        
        # Add noise directly without rounding
        noise = self._sample_geometric_noise(n)
        noisy_values = values + noise
        
        # Find the maximum
        selected_index = int(np.argmax(noisy_values))
        selected_value = float(noisy_values[selected_index])
        
        return selected_index, selected_value

class SecureAggregation:
    """
    Simulates a Secure Aggregation mechanism (simplified model).
    """
    
    def __init__(self, epsilon: float, num_parties: int = 3):
        """
        Initialize the Secure Aggregation Mechanism.
        
        Args:
            epsilon: Privacy parameter (ε)
            num_parties: Number of parties in the protocol
        """
        self.epsilon = epsilon
        self.num_parties = num_parties
        
    def select(self, values: np.ndarray, **kwargs) -> Tuple[int, float]:
        """
        Select an index using secure aggregation simulation.
        
        Args:
            values: Input data vector
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (selected_index, score)
        """
        n = len(values)
        
        # Simulate local noise addition from each party
        noisy_values = values.copy()
        for _ in range(self.num_parties):
            local_epsilon = self.epsilon / self.num_parties
            p = 1 - np.exp(-local_epsilon)
            noise = np.random.geometric(p, size=n) - 1
            noisy_values += noise
        
        # Simulate aggregation and selection
        selected_index = int(np.argmax(noisy_values))
        
        return selected_index, float(values[selected_index]) 