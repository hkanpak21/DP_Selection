import unittest
import numpy as np
import sys
import os
from unittest.mock import patch

# Add the project root directory to the path (parent directory of tests)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.distributed import (
    IntegerSecretSharing, 
    SecureNoiseGeneration,
    SecureArgmax, 
    DistributedDPSelection
)
from src.utils import compute_error_metrics

class TestIntegerSecretSharing(unittest.TestCase):
    """
    Unit tests for the integer secret sharing scheme.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_parties = 3
        self.field_size = 2**16
        self.sharing = IntegerSecretSharing(self.num_parties, self.field_size)
        
    def test_initialization(self):
        """Test that the scheme initializes correctly."""
        self.assertEqual(self.sharing.num_parties, self.num_parties)
        self.assertEqual(self.sharing.field_size, self.field_size)
        
    def test_share_reconstruct_value(self):
        """Test sharing and reconstructing a single value."""
        value = 42
        shares = self.sharing.share(value)
        
        # Check that we have the right number of shares
        self.assertEqual(len(shares), self.num_parties)
        
        # Reconstruct
        reconstructed = self.sharing.reconstruct(shares)
        self.assertEqual(reconstructed, value)
        
    def test_share_reconstruct_vector(self):
        """Test sharing and reconstructing a vector."""
        vector = np.array([1, 2, 3, 4, 5])
        shares = self.sharing.share_vector(vector)
        
        # Check that we have the right number of share vectors
        self.assertEqual(len(shares), self.num_parties)
        
        # Check that each share vector has the right length
        self.assertEqual(len(shares[0]), len(vector))
        
        # Reconstruct
        reconstructed = self.sharing.reconstruct_vector(shares)
        np.testing.assert_array_equal(reconstructed, vector)

class TestSecureNoiseGeneration(unittest.TestCase):
    """
    Unit tests for secure noise generation.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.epsilon = 1.0
        self.num_parties = 3
        self.field_size = 2**16
        self.noise_gen = SecureNoiseGeneration(self.epsilon, self.num_parties, self.field_size)
        
    def test_initialization(self):
        """Test that the noise generator initializes correctly."""
        self.assertEqual(self.noise_gen.epsilon, self.epsilon)
        self.assertEqual(self.noise_gen.num_parties, self.num_parties)
        
    def test_local_noise_sampling(self):
        """Test local noise sampling."""
        size = 10
        noise = self.noise_gen._local_geometric_sample(size)
        
        self.assertEqual(len(noise), size)
        self.assertGreaterEqual(np.min(noise), 0)
        
    def test_distributed_noise(self):
        """Test distributed noise generation."""
        size = 5
        party_noise = [self.noise_gen.generate_local_noise_shares(size) for _ in range(self.num_parties)]
        
        aggregated = self.noise_gen.distributed_geometric_noise(party_noise)
        
        self.assertEqual(len(aggregated), size)

class TestSecureArgmax(unittest.TestCase):
    """
    Unit tests for secure argmax protocol.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_parties = 3
        self.field_size = 2**16
        self.argmax = SecureArgmax(self.num_parties, self.field_size)
        self.sharing = IntegerSecretSharing(self.num_parties, self.field_size)
        
    def test_initialization(self):
        """Test that the protocol initializes correctly."""
        self.assertEqual(self.argmax.num_parties, self.num_parties)
        self.assertEqual(self.argmax.field_size, self.field_size)
        
    def test_secure_compare(self):
        """Test secure comparison."""
        a_val = 10
        b_val = 5
        
        a_shares = self.sharing.share(a_val)
        b_shares = self.sharing.share(b_val)
        
        result = self.argmax._secure_compare(a_shares, b_shares)
        self.assertEqual(result, 1)
        
        # Test with b > a
        result = self.argmax._secure_compare(b_shares, a_shares)
        self.assertEqual(result, 0)
        
    def test_compute_argmax(self):
        """Test computing argmax of shared values."""
        # Set a fixed seed for reproducibility
        np.random.seed(42)
        
        # Create a test vector where the maximum is at index 1
        test_vector = np.array([3, 7, 1, 5])
        
        # Share the vector
        value_shares = self.sharing.share_vector(test_vector)
        
        # Compute the argmax
        max_index_shares = self.argmax.compute_argmax(value_shares)
        
        # Reconstruct the result
        max_index = self.sharing.reconstruct(max_index_shares)
        
        # The maximum value 7 is at index 1
        self.assertEqual(max_index, 1, f"Expected max index 1, got {max_index}")

class TestDistributedDPSelection(unittest.TestCase):
    """
    Integration tests for the distributed DP selection protocol.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.epsilon = 1.0
        self.num_parties = 3
        self.field_size = 2**16
        self.selection = DistributedDPSelection(self.epsilon, self.num_parties, self.field_size)
        
    def test_initialization(self):
        """Test that the protocol initializes correctly."""
        self.assertEqual(self.selection.epsilon, self.epsilon)
        self.assertEqual(self.selection.num_parties, self.num_parties)
        self.assertEqual(self.selection.field_size, self.field_size)
        
    def test_simulate_protocol(self):
        """Test simulating the complete protocol."""
        np.random.seed(42)  # For reproducibility
        
        # Create test vector
        input_vector = np.array([0.2, 0.8, 0.4, 0.6])
        
        # Run simulation
        results = self.selection.simulate_protocol(input_vector, bits=3)
        
        # Check result structure
        self.assertIn("selected_index", results)
        self.assertIn("selected_value", results)
        self.assertIn("true_max_index", results)
        self.assertIn("true_max_value", results)
        self.assertIn("correct_selection", results)
        self.assertIn("value_error", results)
        self.assertIn("utility", results)
        
        # The true max should be at index 1 (value 0.8)
        self.assertEqual(results["true_max_index"], 1)

if __name__ == "__main__":
    unittest.main() 