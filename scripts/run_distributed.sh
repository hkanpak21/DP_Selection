#!/bin/bash
# Script to run distributed model experiments

# Change to project root directory
cd "$(dirname "$0")/.."

# Create data directory if it doesn't exist
mkdir -p data/dpbench

# Generate synthetic data if needed
python -c "
import os
import numpy as np
from src.utils import DataLoader
loader = DataLoader()
datasets = loader.create_benchmark_datasets(sizes=[10, 50, 100], distributions=['uniform', 'normal'], num_samples=2)
for name, data in datasets.items():
    np.save(f'data/dpbench/{name}.npy', data)
print(f'Created {len(datasets)} synthetic datasets for benchmarking')
"

# Run distributed model tests
python tests/test_distributed_dp.py

# Run distributed model benchmarks with smaller parameter space
python tests/benchmark.py --distributed --output results_distributed.json --trials 10

echo "Distributed model experiments completed. Results saved to results_distributed.json" 