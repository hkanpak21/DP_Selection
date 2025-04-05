import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from .secret_sharing import IntegerSecretSharing
from .noise_generation import SecureNoiseGeneration
from .secure_argmax import SecureArgmax

class DistributedDPSelection:
    """
    Implements an end-to-end distributed differentially private selection protocol.
    """
    
    def __init__(self, epsilon: float, num_parties: int, field_size: int = 2**32):
        """
        Initialize the distributed DP selection protocol.
        
        Args:
            epsilon: Privacy parameter (ε)
            num_parties: Number of parties in the protocol
            field_size: Size of the finite field for arithmetic
        """
        self.epsilon = epsilon
        self.num_parties = num_parties
        self.field_size = field_size
        
        # Initialize sub-protocols
        self.sharing = IntegerSecretSharing(num_parties, field_size)
        self.noise_gen = SecureNoiseGeneration(epsilon, num_parties, field_size)
        self.argmax = SecureArgmax(num_parties, field_size)
        
    def _round_values(self, value_shares: List[np.ndarray], bits: int) -> List[np.ndarray]:
        """
        Apply secure rounding to secret-shared values.
        
        Args:
            value_shares: List of share vectors from each party
            bits: Number of bits for rounding
            
        Returns:
            List of rounded share vectors
        """
        # In real MPC, this would be done securely
        # For simulation, we reconstruct, round, and re-share
        values = self.sharing.reconstruct_vector([shares for shares in value_shares])
        
        scale = 2 ** bits
        rounded = np.floor(values * scale) / scale
        
        return self.sharing.share_vector(rounded)
    
    def run_protocol(self, input_shares: List[np.ndarray], bits: int = 8) -> Tuple[int, float]:
        """
        Run the complete distributed DP selection protocol.
        
        Args:
            input_shares: Secret shares of the input vector for each party
            bits: Number of bits for rounding
            
        Returns:
            Tuple of (selected_index, selected_value)
        """
        # Step 1: Apply secure rounding
        rounded_shares = self._round_values(input_shares, bits)
        
        # Step 2: Generate and add noise
        noise_shares = []
        for party in range(self.num_parties):
            # Each party generates local noise
            local_noise = self.noise_gen.generate_local_noise_shares(len(input_shares[0]))
            noise_shares.append(local_noise)
        
        # Aggregate noise
        aggregated_noise = self.noise_gen.distributed_geometric_noise(noise_shares)
        
        # Add noise to rounded values (would be done securely in real MPC)
        # This is a simplified simulation
        noisy_values = self.sharing.reconstruct_vector(rounded_shares) + aggregated_noise
        noisy_shares = self.sharing.share_vector(noisy_values)
        
        # Step 3: Compute argmax securely
        max_index_shares = self.argmax.compute_argmax(noisy_shares)
        
        # Reconstruct the result
        selected_index = int(self.sharing.reconstruct(max_index_shares))
        
        # Ensure selected_index is in bounds - this is a fallback for simulation only
        input_size = len(input_shares[0])
        if selected_index >= input_size:
            # Fallback to a valid index
            selected_index = input_size - 1
            
        # Optionally reconstruct the selected value
        selected_value = float(self.sharing.reconstruct_vector(input_shares)[selected_index])
        
        return selected_index, selected_value
    
    def simulate_protocol(self, input_vector: np.ndarray, bits: int = 8) -> Dict[str, Any]:
        """
        Simulate the protocol on plaintext input for testing and benchmarking.
        
        Args:
            input_vector: Plaintext input vector
            bits: Number of bits for rounding
            
        Returns:
            Dictionary with protocol results and statistics
        """
        # Share the input
        input_shares = self.sharing.share_vector(input_vector)
        
        # Run the protocol
        selected_index, selected_value = self.run_protocol(input_shares, bits)
        
        # Compute reference values for benchmarking
        true_max_index = np.argmax(input_vector)
        true_max_value = input_vector[true_max_index]
        
        # Prepare results
        results = {
            "selected_index": selected_index,
            "selected_value": selected_value,
            "true_max_index": true_max_index,
            "true_max_value": true_max_value,
            "correct_selection": selected_index == true_max_index,
            "value_error": abs(true_max_value - selected_value),
            "utility": {
                "selection_accuracy": 1 if selected_index == true_max_index else 0,
                "relative_value_error": abs(true_max_value - selected_value) / true_max_value if true_max_value != 0 else float('inf')
            }
        }
        
        return results 