import unittest
import numpy as np
import sys
import os

# Add the project root directory to the path (parent directory of tests)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.central import NoiseAndRound, DPSelection
from src.utils import compute_error_metrics

class TestNoiseAndRound(unittest.TestCase):
    """
    Unit tests for the Noise-and-round algorithm.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.epsilon = 1.0
        self.mechanism = NoiseAndRound(epsilon=self.epsilon)
        
    def test_initialization(self):
        """Test that the mechanism initializes correctly."""
        self.assertEqual(self.mechanism.epsilon, self.epsilon)
        self.assertIsNone(self.mechanism.delta)
        
    def test_geometric_noise(self):
        """Test geometric noise sampling."""
        n = 1000
        noise = self.mechanism._sample_geometric_noise(n)
        self.assertEqual(len(noise), n)
        self.assertGreaterEqual(np.min(noise), 0)
        
    def test_rounding(self):
        """Test value rounding."""
        values = np.array([0.12345, 0.67890, 0.3333])
        bits = 3
        rounded = self.mechanism._round_values(values, bits)
        
        # With 3 bits, we should round to multiples of 0.125 (1/8)
        self.assertEqual(rounded[0], 0.125)  # 0.12345 rounds to 0.125
        self.assertEqual(rounded[1], 0.625)  # 0.67890 rounds to 0.625
        self.assertEqual(rounded[2], 0.25)   # 0.3333 rounds to 0.25
        
    def test_selection(self):
        """Test selection of the maximum entry."""
        # Fixed seed for reproducibility
        np.random.seed(42)
        
        values = np.array([0.2, 0.8, 0.4, 0.6])
        index, value = self.mechanism.select(values, bits=3)
        
        # Due to randomness, we can't check exact values, but we can verify types
        self.assertIsInstance(index, int)
        self.assertIsInstance(value, float)
        self.assertGreaterEqual(index, 0)
        self.assertLess(index, len(values))

class TestDPSelection(unittest.TestCase):
    """
    Integration tests for the DP selection mechanism.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.epsilon = 1.0
        self.selection = DPSelection(mechanism="noise_and_round", epsilon=self.epsilon)
        
    def test_initialization(self):
        """Test that the selection mechanism initializes correctly."""
        self.assertEqual(self.selection.mechanism_type, "noise_and_round")
        self.assertIn("epsilon", self.selection.params)
        self.assertEqual(self.selection.params["epsilon"], self.epsilon)
        
    def test_select(self):
        """Test selection function."""
        # Fixed seed for reproducibility
        np.random.seed(42)
        
        data = np.array([0.1, 0.9, 0.5, 0.3])
        index, value = self.selection.select(data, bits=3)
        
        self.assertIsInstance(index, int)
        self.assertIsInstance(value, float)
        
    def test_error_evaluation(self):
        """Test error evaluation functionality."""
        # Fixed seed for reproducibility
        np.random.seed(42)
        
        data = np.array([0.1, 0.9, 0.5, 0.3])
        true_max_index = 1  # Index of max value (0.9)
        
        results = self.selection.evaluate_error(data, true_max_index, num_trials=10)
        
        self.assertIn("accuracy", results)
        self.assertIn("average_error", results)
        self.assertIn("trials", results)
        self.assertEqual(len(results["trials"]), 10)

if __name__ == "__main__":
    unittest.main() 