# Design Document: DP Selection Mechanism

## Architecture

This document outlines the technical design and architecture of the differentially private selection mechanism implementation.

### Central Model

The central model implements the Noise-and-round algorithm as described in the Selection paper. Key components include:

- **Noise Calibration**: Implementation of one-sided geometric or negative binomial distributions
- **Rounding**: Strategies for rounding with privacy preservation
- **Error Analysis**: Methods to validate error bounds

### Distributed MPC Model

The distributed model extends the central approach to a secure multi-party computation setting:

- **Integer Secret Sharing**: Protocol for distributing vector elements among multiple servers
- **Secure Noise Generation**: Methods for generating and aggregating noise in a distributed manner
- **Secure Argmax**: Tree-based comparison protocol with correlated randomness

## Implementation Details

### Privacy Parameters

- ε (epsilon): Privacy budget parameter
- δ (delta): Approximate DP relaxation parameter (when applicable)
- Rounding bits: Parameter controlling utility-privacy tradeoff

### Protocols

- **Security Model**: Semi-honest with honest majority
- **Communication Complexity**: Analysis of communication rounds and total data transfer
- **Computational Complexity**: Analysis of local computation requirements

## Testing Strategy

- Unit tests for individual components
- Integration tests for end-to-end protocol validation
- Benchmarking on both synthetic and real-world datasets 