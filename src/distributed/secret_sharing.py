import numpy as np
from typing import List, Tuple, Dict, Optional

class IntegerSecretSharing:
    """
    Implements integer secret sharing for secure multi-party computation.
    """
    
    def __init__(self, num_parties: int, field_size: int = 2**32):
        """
        Initialize the integer secret sharing scheme.
        
        Args:
            num_parties: Number of parties in the protocol
            field_size: Size of the finite field for arithmetic
        """
        self.num_parties = num_parties
        self.field_size = field_size
        
    def share(self, value: int) -> List[int]:
        """
        Split a value into shares for the parties.
        
        Args:
            value: The value to be secret shared
            
        Returns:
            List of shares, one for each party
        """
        shares = np.random.randint(0, self.field_size, size=self.num_parties-1)
        last_share = (value - np.sum(shares)) % self.field_size
        shares = np.append(shares, last_share)
        
        return shares.tolist()
    
    def share_vector(self, vector: np.ndarray) -> List[np.ndarray]:
        """
        Share each element of a vector among the parties.
        
        Args:
            vector: Vector to be shared
            
        Returns:
            List of share vectors, one for each party
        """
        n = len(vector)
        party_shares = [np.zeros(n, dtype=int) for _ in range(self.num_parties)]
        
        for i in range(n):
            shares = self.share(int(vector[i]))
            for j in range(self.num_parties):
                party_shares[j][i] = shares[j]
                
        return party_shares
    
    def reconstruct(self, shares: List[int]) -> int:
        """
        Reconstruct the original value from its shares.
        
        Args:
            shares: List of shares
            
        Returns:
            The reconstructed value
        """
        return sum(shares) % self.field_size
    
    def reconstruct_vector(self, share_vectors: List[np.ndarray]) -> np.ndarray:
        """
        Reconstruct a vector from its distributed shares.
        
        Args:
            share_vectors: List of share vectors from each party
            
        Returns:
            The reconstructed vector
        """
        n = len(share_vectors[0])
        result = np.zeros(n, dtype=int)
        
        for i in range(n):
            shares = [share_vector[i] for share_vector in share_vectors]
            result[i] = self.reconstruct(shares)
            
        return result 