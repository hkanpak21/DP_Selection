#!/bin/bash
# Script to run central model experiments

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
datasets = loader.create_benchmark_datasets(sizes=[10, 100, 1000], distributions=['uniform', 'normal', 'exponential'], num_samples=3)
for name, data in datasets.items():
    np.save(f'data/dpbench/{name}.npy', data)
print(f'Created {len(datasets)} synthetic datasets for benchmarking')
"

# Run central model tests
python tests/test_central_dp.py

# Run central model benchmarks
python tests/benchmark.py --central --output results_central.json

echo "Central model experiments completed. Results saved to results_central.json" 