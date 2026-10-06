# AGENTS.md — Repository Knowledge for AI Assistants

## Project Overview
Geometric multi-skew Brownian motion (MSBM) optimal stopping experiments.
All code is SYNTHETIC (no market data). Based on optimal stopping theory for skew diffusions.

## Repository Structure
- `src/`: Core mathematical modules (harmonic functions, LCM, exit/entry problems)
- `exp/`: Experiment orchestration scripts (exp_1 through exp_4)
- `tests/`: pytest tests for all src modules
- `results/`: Output tables and plots; RESULTS.md summarizes findings

## Key Mathematical Objects
- **Roots** α₁ < 0 < 1 < α₂: from ODE 0.5σ²α(α-1) + μα - r = 0
- **κ₁, κ₂**: kappa constants derived from roots
- **T_β(z)**: 2×2 crossing matrix for skew level z with coefficient β
- **ψ_r, φ_r**: increasing/decreasing r-harmonic functions (piecewise power law)
- **F = ψ_r/φ_r**: strictly increasing F-transform
- **LCM**: Least Concave Majorant via upper convex hull algorithm
- **Exit problem**: v(x) = φ_r(x) · LCM(g/φ_r over F)
- **Entry problem**: u(x) = φ_r(x) · LCM(W/φ_r over F), W = v - x

## Parameters Used
- Base: r=0.05, μ=0.01, σ=0.2, L=1
- Grid: 400000 points geometric spacing on [1e-4, 1e4]
- Numerical tolerance: tol_contact=1e-10

## No Market Data
This project uses ONLY synthetic model inputs. No Massive API calls needed.

## Testing
Run `pytest tests/ -v` from project root.
All tests use synthetic data and deterministic computations.

## Experiment Scripts (`exp/`)
Four experiment files, all runnable as `python -m exp.exp_XXX`:

| File | Regime | Key result |
|------|--------|-----------|
| `exp_support_resistance.py` | β_L>0>β_H | P+={L}, a/b/W(L) match expected |
| `exp_two_support.py` | β_L>0, β_H>0 | P+={L,H}, ratio-window classifies contact |
| `exp_other_regimes.py` | both neg / lower-neg-upper-pos | Case A: W≈0; Case B: P+={H}, hitting-time formula |
| `exp_monotonicity.py` | single-skew β∈[0.05,0.95] | w(β) strictly increasing, w(0.05)≈6.2e-4, w(0.95)≈0.202 |

## Important Implementation Notes

### Grid precision
`np.geomspace(1e-4, 1e4, 400_000)` does NOT include x=1.0 exactly (requires N odd = 400001).
This causes ~2e-5 discrepancy between `eval_at_point` (exact) and grid interpolation.
All a/b/W values still match expected to 4 decimal places.

### W(x) at level points
Use `eval_at_point(x, α₁, α₂, coeffs)` + `np.interp(F_x, F, K_g)` for exact values:
```python
phi_L = eval_at_point(L, alpha_1, alpha_2, varphi_coeffs)
F_L   = eval_at_point(L, alpha_1, alpha_2, psi_coeffs) / phi_L
W_L   = phi_L * np.interp(F_L, F, K_g) - L
```
Note: `phi_r(L) = 1` always (normalization), `psi_r(L) = 1` for level L=1 (bottom of single level).

### G₁ formula (Exp 4)
`G₁(m) = κ₁·m^{1-α₁} + κ₂·m^{1-α₂}` gives v(1)=G₁(ā) analytically from smooth-pasting
at the lower exit boundary ā in the standard-GBM limit; holds also for single-skew at L=1
because φ_r(1)=1 and ψ_r(1)=1.

### Contact set for two-level entry (Exp 2)
`classify_entry_contact` correctly identifies {H} (rho<underline), {L,H} (underline≤rho≤overline)
but **incorrectly returns {L,H} when rho>overline** because upper_hull_indices always includes
the rightmost point. Solution: use `compute_ratio_window`'s `predicted` field to select
support points for `solve_entry_theorem`.

### Key function signatures
- `build_harmonic_functions(r, mu, sigma, levels, betas, x_grid)` → dict
- `solve_exit(x_grid, g, varphi_r, F)` → dict {v, K_g, contact_mask}
- `extract_exit_band(x_grid, v, g)` → (a, b)
- `solve_entry_theorem(x_grid, W_support, phi_support, F_support, varphi_r_grid, F_grid)` → array
- `solve_entry_bruteforce(x_grid, W, varphi_r, F)` → dict {u, K_W, contact_mask}
- `compute_hitting_time_formula(x_grid, H_level, W_H, psi_r, psi_H, varphi_r, phi_H)` → array
