#!/bin/bash
# Script to run comprehensive benchmarks for both central and distributed models

# Change to project root directory
cd "$(dirname "$0")/.."

# Create data directory if it doesn't exist
mkdir -p data/dpbench

# Generate synthetic data for benchmarking
python -c "
import os
import numpy as np
from src.utils import DataLoader
loader = DataLoader()

# Generate datasets of different sizes and distributions
datasets = loader.create_benchmark_datasets(
    sizes=[10, 50, 100, 500],
    distributions=['uniform', 'normal', 'exponential'],
    num_samples=2
)

# Save datasets
for name, data in datasets.items():
    np.save(f'data/dpbench/{name}.npy', data)
    
print(f'Created {len(datasets)} synthetic datasets for benchmarking')
"

# Create results directory
mkdir -p results

# Run all tests to ensure functionality
echo "Running all tests..."
python -m unittest discover tests

# Run benchmarks for both models
echo "Running comprehensive benchmarks..."
python tests/benchmark.py --central --distributed --output results/benchmarks_complete.json --trials 20

# Generate comparison plots
python -c "
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# Load benchmark results
with open('results/benchmarks_complete.json', 'r') as f:
    results = json.load(f)
    
# Convert to DataFrame
all_results = results['central'] + results['distributed']
df = pd.DataFrame(all_results)

# Create results directory if it doesn't exist
os.makedirs('results/plots', exist_ok=True)

# Plot accuracy vs epsilon
plt.figure(figsize=(10, 6))
for model in ['central', 'distributed']:
    model_data = df[df['model'] == model]
    grouped = model_data.groupby('epsilon')['accuracy'].mean()
    plt.plot(grouped.index, grouped.values, marker='o', label=model)
    
plt.xlabel('Privacy parameter (ε)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Privacy Parameter')
plt.legend()
plt.grid(True)
plt.savefig('results/plots/accuracy_vs_epsilon.png')

# Plot error vs bits
plt.figure(figsize=(10, 6))
for model in ['central', 'distributed']:
    model_data = df[df['model'] == model]
    grouped = model_data.groupby('bits')['average_error'].mean()
    plt.plot(grouped.index, grouped.values, marker='o', label=model)
    
plt.xlabel('Rounding bits')
plt.ylabel('Average error')
plt.title('Error vs Rounding Precision')
plt.legend()
plt.grid(True)
plt.savefig('results/plots/error_vs_bits.png')

# Plot time performance
plt.figure(figsize=(10, 6))
for model in ['central', 'distributed']:
    model_data = df[df['model'] == model]
    plt.scatter(model_data['epsilon'], model_data['time_per_trial'], label=model, alpha=0.7)
    
plt.xlabel('Privacy parameter (ε)')
plt.ylabel('Time per trial (seconds)')
plt.title('Performance Comparison')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.savefig('results/plots/performance_comparison.png')

print('Generated comparison plots in results/plots/')
"

echo "Benchmarks completed. Results saved to results/benchmarks_complete.json and plots to results/plots/" 