import sys
import os
import numpy as np
import pandas as pd
import time
import json
import argparse
from typing import Dict, List, Tuple, Any

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.central import DPSelection
from src.distributed import DistributedDPSelection
from src.utils import DataLoader, compute_error_metrics

class DPSelectionBenchmark:
    """
    Benchmark utility for DP selection mechanisms.
    """
    
    def __init__(self, data_dir: str = "data/dpbench"):
        """
        Initialize the benchmark.
        
        Args:
            data_dir: Directory containing datasets
        """
        self.data_loader = DataLoader(data_dir)
        self.results = {
            "central": [],
            "distributed": []
        }
        
    def benchmark_central(self, epsilons: List[float] = [0.1, 0.5, 1.0, 2.0], 
                        bits_values: List[int] = [3, 5, 8],
                        num_trials: int = 50) -> List[Dict[str, Any]]:
        """
        Benchmark the central model.
        
        Args:
            epsilons: List of privacy parameters to test
            bits_values: List of rounding bit values to test
            num_trials: Number of trials per configuration
            
        Returns:
            List of benchmark results
        """
        central_results = []
        
        # Generate synthetic datasets
        datasets = self.data_loader.create_benchmark_datasets()
        
        for dataset_name, data in datasets.items():
            true_max_index = np.argmax(data)
            
            for epsilon in epsilons:
                for bits in bits_values:
                    # Create selection mechanism
                    selection = DPSelection(mechanism="noise_and_round", epsilon=epsilon)
                    
                    # Run trials
                    start_time = time.time()
                    
                    results = selection.evaluate_error(data, true_max_index, num_trials=num_trials)
                    
                    elapsed_time = time.time() - start_time
                    
                    central_results.append({
                        "dataset": dataset_name,
                        "epsilon": epsilon,
                        "bits": bits,
                        "accuracy": results["accuracy"],
                        "average_error": results["average_error"],
                        "time_per_trial": elapsed_time / num_trials,
                        "num_trials": num_trials,
                        "model": "central"
                    })
                    
                    print(f"Central benchmark: dataset={dataset_name}, epsilon={epsilon}, bits={bits}, accuracy={results['accuracy']:.4f}")
        
        self.results["central"] = central_results
        return central_results
    
    def benchmark_distributed(self, epsilons: List[float] = [0.1, 0.5, 1.0, 2.0],
                           bits_values: List[int] = [3, 5, 8],
                           num_parties_values: List[int] = [3, 5],
                           num_trials: int = 20) -> List[Dict[str, Any]]:
        """
        Benchmark the distributed model.
        
        Args:
            epsilons: List of privacy parameters to test
            bits_values: List of rounding bit values to test
            num_parties_values: List of party counts to test
            num_trials: Number of trials per configuration
            
        Returns:
            List of benchmark results
        """
        distributed_results = []
        
        # Generate synthetic datasets
        datasets = self.data_loader.create_benchmark_datasets(sizes=[10, 50, 100])
        
        for dataset_name, data in datasets.items():
            true_max_index = np.argmax(data)
            
            for epsilon in epsilons:
                for bits in bits_values:
                    for num_parties in num_parties_values:
                        # Create selection protocol
                        selection = DistributedDPSelection(epsilon, num_parties)
                        
                        # Run trials
                        accuracies = []
                        errors = []
                        times = []
                        
                        for _ in range(num_trials):
                            start_time = time.time()
                            
                            results = selection.simulate_protocol(data, bits)
                            
                            elapsed_time = time.time() - start_time
                            times.append(elapsed_time)
                            
                            accuracies.append(results["utility"]["selection_accuracy"])
                            errors.append(results["value_error"])
                        
                        distributed_results.append({
                            "dataset": dataset_name,
                            "epsilon": epsilon,
                            "bits": bits,
                            "num_parties": num_parties,
                            "accuracy": np.mean(accuracies),
                            "average_error": np.mean(errors),
                            "time_per_trial": np.mean(times),
                            "num_trials": num_trials,
                            "model": "distributed"
                        })
                        
                        print(f"Distributed benchmark: dataset={dataset_name}, epsilon={epsilon}, bits={bits}, parties={num_parties}, accuracy={np.mean(accuracies):.4f}")
        
        self.results["distributed"] = distributed_results
        return distributed_results
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare the central and distributed models.
        
        Returns:
            DataFrame with comparison results
        """
        all_results = self.results["central"] + self.results["distributed"]
        results_df = pd.DataFrame(all_results)
        
        return results_df
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """
        Save benchmark results to a file.
        
        Args:
            filename: Output filename
        """
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
            
    def load_results(self, filename: str = "benchmark_results.json"):
        """
        Load benchmark results from a file.
        
        Args:
            filename: Input filename
        """
        with open(filename, "r") as f:
            self.results = json.load(f)

def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Run DP selection benchmarks")
    parser.add_argument("--central", action="store_true", help="Run central model benchmarks")
    parser.add_argument("--distributed", action="store_true", help="Run distributed model benchmarks")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file for results")
    parser.add_argument("--data-dir", default="data/dpbench", help="Data directory")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials per configuration")
    
    args = parser.parse_args()
    
    benchmark = DPSelectionBenchmark(args.data_dir)
    
    if args.central:
        benchmark.benchmark_central(num_trials=args.trials)
        
    if args.distributed:
        benchmark.benchmark_distributed(num_trials=args.trials)
        
    if args.central or args.distributed:
        benchmark.save_results(args.output)
        
        # Generate comparison if both models were benchmarked
        if args.central and args.distributed:
            comparison = benchmark.compare_models()
            print("\nModel Comparison Summary:")
            print(comparison.groupby(['model', 'epsilon']).mean())
    else:
        print("No benchmarks specified. Use --central or --distributed to run benchmarks.")

if __name__ == "__main__":
    main() 