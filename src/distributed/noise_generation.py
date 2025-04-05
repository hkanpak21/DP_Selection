import numpy as np
from typing import List, Tuple, Optional
import math

class SecureNoiseGeneration:
    """
    Implements secure noise generation for distributed differentially private computations.
    """
    
    def __init__(self, epsilon: float, num_parties: int, field_size: int = 2**32):
        """
        Initialize the secure noise generation protocol.
        
        Args:
            epsilon: Privacy parameter (ε)
            num_parties: Number of parties in the protocol
            field_size: Size of the finite field for arithmetic
        """
        self.epsilon = epsilon
        self.num_parties = num_parties
        self.field_size = field_size
        
    def _local_geometric_sample(self, size: int = 1) -> np.ndarray:
        """
        Generate noise samples from a one-sided geometric distribution locally.
        
        Args:
            size: Number of samples to generate
            
        Returns:
            Array of noise samples
        """
        # Local parameter adjustment for distributed setting
        # Each party uses ε/num_parties for local noise
        local_epsilon = self.epsilon / self.num_parties
        p = 1 - np.exp(-local_epsilon)
        
        return np.random.geometric(p, size=size) - 1
    
    def generate_local_noise_shares(self, size: int = 1) -> np.ndarray:
        """
        Generate local noise shares for distributed noise protocol.
        
        Args:
            size: Number of noise values to generate
            
        Returns:
            Local noise shares
        """
        return self._local_geometric_sample(size)
    
    def aggregate_noise(self, noise_shares: List[np.ndarray]) -> np.ndarray:
        """
        Securely aggregate noise shares from all parties.
        
        Args:
            noise_shares: List of noise shares from all parties
            
        Returns:
            Aggregated noise values
        """
        # In a real implementation, this would involve secure MPC
        # For simulation purposes, we simply sum the shares
        return np.sum(noise_shares, axis=0) % self.field_size
    
    def distributed_geometric_noise(self, parties_noise: List[np.ndarray]) -> np.ndarray:
        """
        Perform the complete distributed noise generation protocol.
        
        Args:
            parties_noise: Noise contributions from each party
            
        Returns:
            Final noise values with DP guarantees
        """
        # Ensure we have the right number of parties
        if len(parties_noise) != self.num_parties:
            raise ValueError(f"Expected {self.num_parties} parties, got {len(parties_noise)}")
        
        # Aggregate the noise shares
        aggregated_noise = self.aggregate_noise(parties_noise)
        
        return aggregated_noise 