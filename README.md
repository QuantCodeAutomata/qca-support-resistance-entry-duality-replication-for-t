# Support–Resistance Entry Duality: Geometric Multi-Skew Brownian Motion

This repository implements and validates the optimal stopping theory for geometric multi-skew Brownian motion (MSBM) as described in the paper on support–resistance entry duality.

## Overview

The project numerically validates:
- **Exp 1**: Support–Resistance Entry Duality (β_L > 0 > β_H)
- **Exp 2**: Two-Support Entry Classification (β_L > 0, β_H > 0)
- **Exp 3**: Other Sign-Pair Regimes (all-negative and mixed-sign skews)
- **Exp 4**: Single-Skew Premium Monotonicity Study

## Mathematical Model

Geometric multi-skew Brownian motion with two skew levels {L, H}:

    dS_t = μ S_t dt + σ S_t dW_t + (β_L/L) S_t dℓ_t^L(S) + (β_H/H) S_t dℓ_t^H(S)

Parameters: r=0.05 (discount), μ=0.01 (drift), σ=0.2 (volatility), L=1 (lower level).

## Repository Structure

```
repo/
├── src/                    # Core mathematical modules
│   ├── roots_and_crossing.py   # ODE roots, crossing matrices
│   ├── harmonic.py             # Piecewise harmonic functions ψ_r, φ_r, F
│   ├── lcm.py                  # Least concave majorant algorithm
│   ├── exit_problem.py         # Numerical exit value computation
│   └── entry_problem.py        # Entry value and contact set
├── exp/                    # Experiment scripts
│   ├── exp_support_resistance.py  # Exp 1
│   ├── exp_two_support.py         # Exp 2
│   ├── exp_other_regimes.py       # Exp 3
│   └── exp_monotonicity.py        # Exp 4
├── tests/                  # pytest tests
├── data/                   # (no market data; synthetic experiments)
├── results/                # Output tables, plots, RESULTS.md
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run all experiments
python -m exp.exp_support_resistance
python -m exp.exp_two_support
python -m exp.exp_other_regimes
python -m exp.exp_monotonicity

# Run tests
pytest tests/ -v
```

## Results

See `results/RESULTS.md` for detailed experimental outputs and comparison with paper values.
