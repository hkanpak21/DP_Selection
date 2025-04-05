# Project Overview

We are replicating the tests and implementing the algorithms described in the "Selection.pdf" paper. Our goal is to build a differentially private (DP) selection mechanism that works in both a central and a distributed (MPC-based) environment. The project will focus on replicating the Noise-and-round mechanism from the central model and extending it to a secure multi-party computation (MPC) protocol that uses integer secret sharing, secure noise generation, and a secure Argmax protocol. This work will be integrated and tested within our existing environment.

---

# Core Functionalities

- **DP Selection Mechanism:**  
  – Implement the Noise-and-round algorithm to select an approximately largest entry in a vector with differential privacy guarantees.  
  – Calibrate noise using one-sided geometric (or negative binomial) distributions with appropriate rounding.

- **Central Model Implementation:**  
  – Develop a standalone DP selection module that operates in the central setting.  
  – Validate privacy guarantees and error bounds through unit tests and benchmarks.

- **Distributed MPC Implementation:**  
  – Implement integer secret sharing for splitting input vectors among multiple servers.  
  – Develop secure noise generation and aggregation modules to compute a DP sum in a distributed setting.  
  – Implement secure, tree-based Argmax computation with pre-generated correlated randomness for efficient comparisons.  
  – Ensure the protocol meets the semi-honest security model with honest majority assumptions.

- **Testing and Benchmarking:**  
  – Create comprehensive tests that replicate the experiments from the paper using both synthetic and real-world DPBench datasets.  
  – Benchmark performance in terms of utility (error measurement), time per computation, and communication overhead.

- **Security and Scalability:**  
  – Verify differential privacy and secure computation properties under varying parameters (ε, rounding bits, number of servers).  
  – Evaluate scalability for high-dimensional vectors and large distributed datasets.

---

# Docs

- **Primary Reference:**  
  – Selection.pdf (the project paper with detailed algorithms and proofs)

- **State-of-the-Art Repositories:**  
  – [MP-SPDZ Repository](https://github.com/data61/MP-SPDZ) – for MPC protocol implementations and secure computation frameworks.  
  – [DPBench Repository](https://github.com/google/dpbench) – for datasets and benchmarking differentially private algorithms.  
  – (Optional) Additional repositories such as the "Permute-and-flip" implementation if further baseline comparisons are needed.

---

# Current File Structure

Below is a proposed file structure to implement the core functionalities:

```
Project-Name/
├── README.md                    # Overview and setup instructions
├── docs/
│   ├── Selection.pdf            # Paper with algorithms and technical details
│   └── design_document.md       # Detailed design and architecture documentation
├── src/
│   ├── central/                 # Central model implementation
│   │   ├── noise_and_round.py   # Implements Noise-and-round algorithm (Algorithm 1)
│   │   └── dp_selection.py      # Wrapper module for DP selection tests in central setting
│   ├── distributed/             # Distributed MPC implementation
│   │   ├── secret_sharing.py    # Integer secret sharing functions and utilities
│   │   ├── noise_generation.py  # Local noise sampling and aggregation module
│   │   ├── secure_argmax.py     # Secure, tree-based Argmax protocol implementation
│   │   └── distributed_dp_selection.py  # End-to-end distributed DP selection protocol
│   └── utils/                   # Utility functions (e.g., math, data handling)
│       ├── math_utils.py
│       └── data_loader.py
├── tests/
│   ├── test_central_dp.py       # Unit and integration tests for the central model
│   ├── test_distributed_dp.py   # Unit and integration tests for the distributed model
│   └── benchmark.py             # Benchmark scripts for utility and performance evaluation
├── data/
│   └── dpbench/                 # DPBench datasets for testing and benchmarking
│       ├── dataset1.csv
│       └── dataset2.csv
└── scripts/
    ├── run_central.sh           # Shell script to execute central model experiments
    ├── run_distributed.sh       # Shell script to execute distributed MPC experiments
    └── run_benchmarks.sh        # Script to run comprehensive benchmarks and generate reports
```