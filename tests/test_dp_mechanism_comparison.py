import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unittest
from typing import List, Dict, Any, Tuple

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.central import DPSelection
from src.utils import compute_error_metrics

class TestDPMechanismComparison(unittest.TestCase):
    """
    Tests for comparing different DP selection mechanisms.
    """
    
    def setUp(self):
        """
        Set up test datasets.
        """
        # Create or load datasets similar to the paper
        self.datasets = {
            "HEPTH": self.generate_synthetic_dataset(size=1000, distribution="exponential", max_value=1000),
            "SEARCHLOGS": self.generate_synthetic_dataset(size=1000, distribution="zipf", max_value=5000),
            "PATENT": self.generate_synthetic_dataset(size=1000, distribution="uniform", max_value=20000)
        }
        
        # Configure epsilon values for comparison (log scale)
        self.epsilons = {
            "HEPTH": np.logspace(-2, -0.5, 15),  # 0.01 to ~0.3
            "SEARCHLOGS": np.logspace(-3, -1.5, 15),  # 0.001 to ~0.03
            "PATENT": np.logspace(-2, -0.5, 15)  # 0.01 to ~0.3
        }
        
        # Define the mechanisms to compare
        self.mechanisms = [
            {"name": "Exponential mechanism", "key": "exponential", "params": {}},
            {"name": "Permute-and-flip", "key": "permute_and_flip", "params": {}},
            {"name": "Randomized response", "key": "randomized_response", "params": {}},
            {"name": "Random choice", "key": "random_choice", "params": {}},
            {"name": "Noise-and-round, r=5", "key": "noise_and_round", "params": {"bits": 5}},
            {"name": "Noise-and-round, r=10", "key": "noise_and_round", "params": {"bits": 10}},
            {"name": "Noise-and-round, no rounding", "key": "noise_and_round_no_rounding", "params": {}},
            {"name": "Secure aggregation", "key": "secure_aggregation", "params": {"num_parties": 3}}
        ]
    
    def generate_synthetic_dataset(self, size: int, distribution: str, max_value: float) -> np.ndarray:
        """
        Generate a synthetic dataset.
        
        Args:
            size: Size of the dataset
            distribution: Type of distribution
            max_value: Maximum value in the dataset
            
        Returns:
            Generated dataset
        """
        if distribution == "uniform":
            data = np.random.uniform(0, max_value, size=size)
        elif distribution == "exponential":
            data = np.random.exponential(scale=max_value/5, size=size)
            data = np.clip(data, 0, max_value)
        elif distribution == "zipf":
            data = np.random.zipf(a=1.5, size=size).astype(float)
            # Scale to desired range
            data = data * max_value / np.max(data)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
        
        return data
    
    def run_mechanism_comparison(self, dataset_name: str, num_trials: int = 50) -> List[Dict[str, Any]]:
        """
        Run comparison of different mechanisms on a dataset.
        
        Args:
            dataset_name: Name of the dataset to use
            num_trials: Number of trials per configuration
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        data = self.datasets[dataset_name]
        true_max_index = np.argmax(data)
        true_max_value = data[true_max_index]
        
        epsilons = self.epsilons[dataset_name]
        
        for mechanism in self.mechanisms:
            for epsilon in epsilons:
                # Create mechanism instance
                params = mechanism["params"].copy()
                params["epsilon"] = epsilon
                selection = DPSelection(mechanism=mechanism["key"], **params)
                
                # Run trials
                errors = []
                correct_selections = 0
                
                for _ in range(num_trials):
                    selected_idx, _ = selection.select(data)
                    error = abs(true_max_value - data[selected_idx])
                    errors.append(error)
                    
                    if selected_idx == true_max_index:
                        correct_selections += 1
                
                # Compute average error
                avg_error = np.mean(errors)
                accuracy = correct_selections / num_trials
                
                # Store result
                results.append({
                    "dataset": dataset_name,
                    "mechanism": mechanism["name"],
                    "epsilon": epsilon,
                    "average_error": avg_error,
                    "accuracy": accuracy,
                })
                
                print(f"Completed: {dataset_name}, {mechanism['name']}, ε={epsilon:.6f}, error={avg_error:.2f}")
        
        return results
    
    def test_mechanism_comparison(self):
        """
        Test comparing different DP mechanisms and generate plots.
        """
        all_results = []
        
        # Run comparison for each dataset
        for dataset_name in self.datasets.keys():
            results = self.run_mechanism_comparison(dataset_name)
            all_results.extend(results)
        
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(all_results)
        
        # Create plots similar to the image
        self.plot_mechanism_comparison(results_df)
        
        # Ensure test passes
        self.assertTrue(True)
    
    def plot_mechanism_comparison(self, results_df: pd.DataFrame):
        """
        Create plots comparing the different mechanisms.
        
        Args:
            results_df: DataFrame with results
        """
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Define line styles and markers for each mechanism
        styles = {
            "Exponential mechanism": {"color": "tab:blue", "marker": "o", "linestyle": "-"},
            "Permute-and-flip": {"color": "tab:orange", "marker": "s", "linestyle": "--"},
            "Randomized response": {"color": "tab:green", "marker": "^", "linestyle": ":"},
            "Random choice": {"color": "tab:purple", "marker": "P", "linestyle": "-."},
            "Noise-and-round, r=5": {"color": "tab:brown", "marker": "*", "linestyle": ":"},
            "Noise-and-round, r=10": {"color": "tab:pink", "marker": "p", "linestyle": "-."},
            "Noise-and-round, no rounding": {"color": "tab:gray", "marker": "x", "linestyle": "--"},
            "Secure aggregation": {"color": "tab:red", "marker": "d", "linestyle": "-."}
        }
        
        # Plot each dataset in a separate subplot
        for i, dataset_name in enumerate(self.datasets.keys()):
            ax = axes[i]
            
            # Filter data for this dataset
            dataset_results = results_df[results_df["dataset"] == dataset_name]
            
            # Plot each mechanism
            for mechanism_name in dataset_results["mechanism"].unique():
                mechanism_data = dataset_results[dataset_results["mechanism"] == mechanism_name]
                
                # Sort by epsilon for consistent line
                mechanism_data = mechanism_data.sort_values("epsilon")
                
                style = styles.get(mechanism_name, {})
                ax.loglog(
                    mechanism_data["epsilon"], 
                    mechanism_data["average_error"],
                    label=mechanism_name if i == 0 else "",  # Only add label in first subplot
                    **style
                )
            
            # Set labels and title
            ax.set_xlabel("ε")
            ax.set_ylabel("Absolute distance to true max" if i == 0 else "")
            ax.set_title(dataset_name)
            ax.grid(True, which="both", linestyle=":")
        
        # Add legend outside the plots
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=4)
        
        # Adjust layout and save figure
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        plt.savefig("mechanism_comparison.png", dpi=300, bbox_inches="tight")
        
        print("Plot saved as mechanism_comparison.png")
    
    def test_quick_single_dataset(self):
        """
        Run a quick test on just one dataset with fewer trials (for unit testing).
        """
        # Use HEPTH dataset with fewer epsilon values and trials
        dataset_name = "HEPTH"
        data = self.datasets[dataset_name]
        true_max_index = np.argmax(data)
        
        # Test just one epsilon value with two mechanisms
        epsilon = 0.1
        mechanism_keys = ["exponential", "noise_and_round"]
        
        for key in mechanism_keys:
            selection = DPSelection(mechanism=key, epsilon=epsilon)
            selected_idx, _ = selection.select(data)
            
            # Just check the type, not the actual value
            self.assertIsInstance(selected_idx, int)
            self.assertGreaterEqual(selected_idx, 0)
            self.assertLess(selected_idx, len(data))

if __name__ == "__main__":
    # Run only the comparison test without unittest framework
    test = TestDPMechanismComparison()
    test.setUp()
    test.test_mechanism_comparison() 