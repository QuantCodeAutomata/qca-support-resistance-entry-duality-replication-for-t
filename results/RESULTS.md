# Experimental Results — Multi-Skew BM Optimal Stopping

## Model Parameters (all experiments)
- Discount rate: r = 0.05
- Drift: μ = 0.01
- Volatility: σ = 0.2
- Lower skew level: L = 1
- Grid: N = 400,000 points on [1e-4, 1e4] (geometric spacing)
- Numerical tolerance: tol_contact = 1e-10

## Computed Characteristic Roots
| Parameter | Value |
|-----------|-------|
| α₁ | −1.350781 |
| α₂ | +1.850781 |
| κ₁ | 0.265739 |
| κ₂ | 0.734261 |

---

## Exp 1: Support–Resistance Entry Duality (β_L > 0 > β_H)

**Claim**: With β_L > 0 > β_H, the optimal entry/contact set is exactly {L}.

| Scenario | H | β_L | β_H | a (exit) | b (exit) | W(L) | max\|u_theory − u_bf\| |
|----------|-----|------|------|----------|----------|------|------------------------|
| A | 1.60 | 0.9 | −0.30 | 0.6253 | 1.4653 | 0.182971 | 1.57e-05 |
| B | 1.15 | 0.9 | −0.20 | 0.6683 | 1.3422 | 0.137656 | 1.76e-05 |
| C | 1.15 | 0.9 | −0.50 | 0.7003 | 1.1500 | 0.109211 | 1.92e-05 |

**Paper targets**: A: a≈0.6253, b≈1.4654, W(L)≈0.182971 ✓ | B: a≈0.6683, b≈1.3422, W(L)≈0.137656 ✓ | C: a≈0.7003, b≈1.1500, W(L)≈0.109221 ✓

**Validation**: For all 3 scenarios, upward-skew set P+ = {L}, contact set = {L}, theorem-based and brute-force entry values agree on [0.3, 3] within ~2e-5 (discretization-limited; paper reports ~3e-10 at N=∞).

---

## Exp 2: Two-Support Entry Classification (β_L > 0, β_H > 0)

**Claim**: Entry contact set ∈ {{L}, {H}, {L,H}}, classified by ratio window ρ̲ ≤ ρ ≤ ρ̄.

| H | β_L | β_H | W(L) | W(H) | ρ | ρ̲ | ρ̄ | Predicted | Contact | Match |
|-----|------|------|------|------|------|------|------|-----------|---------|-------|
| 3.0 | 0.9 | 0.9 | 0.182971 | 0.548912 | 0.3333 | 0.2794 | 36.91 | L,H | {1,3} | ✓ |
| 3.0 | 0.5 | 0.9 | 0.060278 | 0.548912 | 0.1098 | 0.2091 | 36.91 | H | {3} | ✓ |
| 3.0 | 0.9 | 0.5 | 0.182971 | 0.180833 | 1.0118 | 0.2794 | 8.022 | L,H | {1,3} | ✓ |
| 1.5 | 0.9 | 0.9 | 0.410305 | 0.293829 | 1.3964 | 0.7845 | 11.28 | L,H | {1,1.5} | ✓ |
| 1.5 | 0.6 | 0.9 | 0.250582 | 0.286157 | 0.8757 | 0.6895 | 11.28 | L,H | {1,1.5} | ✓ |
| 1.3 | 0.9 | 0.3 | 0.254345 | 0.100583 | 2.5287 | 0.8934 | 1.718 | L | {1} | ✓ |
| 2.2 | 0.7 | 0.8 | 0.114627 | 0.323801 | 0.3540 | 0.4135 | 11.91 | H | {2.2} | ✓ |

**Validation**: All 7 contact-set classifications match expected (ratio-window corollary validated numerically). Three distinct outcomes {L}, {H}, {L,H} all observed.

---

## Exp 3: Other Sign-Pair Regimes

### Case A: Both Skews Negative (P+ = ∅ → u ≡ 0)

| H | β_L | β_H | max\|W\| | Contact |
|-----|------|------|----------|---------|
| 1.5 | −0.5 | −0.5 | 8.88e-16 | ∅ |
| 1.3 | −0.9 | −0.3 | 8.88e-16 | ∅ |

Both scenarios: max|W| ≈ 8.9e-16 (machine epsilon), well below the paper's 2e-12 threshold ✓.

### Case B: Lower Negative, Upper Positive (P+ = {H})

| H | β_L | β_H | a (exit) | b (exit) | W(H) | err(hit,bf) | Contact |
|-----|------|------|----------|----------|------|-------------|---------|
| 3.0 | −0.5 | 0.9 | 1.8758 | 4.3959 | 0.548912 | 4.48e-05 | {H} |
| 1.2 | −0.2 | 0.9 | 0.8892 | 1.7504 | 0.213649 | 2.25e-05 | {H} |
| 1.2 | −0.6 | 0.9 | 1.0000 | 1.7486 | 0.212364 | 1.76e-05 | {H} |

**Paper targets**: B1: (1.876, 4.396), W(H)≈0.548912 ✓ | B2: (0.889, 1.750), W(H)≈0.213649 ✓ | B3: (1.000, 1.749), W(H)≈0.212365 ✓

Hitting-time formula u(x) = W(H)·ψ_r(x)/ψ_r(H) [x≤H], W(H)·φ_r(x)/φ_r(H) [x≥H] agrees with brute-force within ~5e-5 on [0.3,6].

---

## Exp 4: Single-Skew Premium Monotonicity

**Claim**: w(β) = G₁(ā(β)) − 1 is strictly increasing in β over {0.05, 0.10, …, 0.95}.

| β | ā(β) | w(β) |
|----|------|------|
| 0.05 | 0.9751 | 6.26e-04 |
| 0.10 | 0.9507 | 2.50e-03 |
| 0.20 | 0.9029 | 9.94e-03 |
| 0.30 | 0.8569 | 2.22e-02 |
| 0.50 | 0.7710 | 6.03e-02 |
| 0.70 | 0.6938 | 1.146e-01 |
| 0.90 | 0.6253 | 1.830e-01 |
| 0.95 | 0.6094 | 2.020e-01 |

**Validation**:
- Strict monotonicity: True (minimum increment = 1.87e-03) ✓
- w(0.05) = 6.26e-04 ≈ paper's 6.2e-04 ✓
- w(0.95) = 0.2020 ≈ paper's 0.202 ✓
- Max consistency error |w_formula − W(1) direct| = 1.56e-05 ✓

---

## Numerical Performance Summary

| Experiment | Key metric | Value |
|------------|-----------|-------|
| Exp 1 | Max theorem vs brute-force error on [0.3,3] | ≤ 1.92e-05 |
| Exp 2 | Contact-set classification accuracy | 7/7 ✓ |
| Exp 3 (Case A) | Max entry value (all-negative) | 8.88e-16 |
| Exp 3 (Case B) | Max hitting-time vs brute-force error on [0.3,6] | ≤ 4.48e-05 |
| Exp 4 | Min monotonicity increment | 1.87e-03 |

> **Note on numerical precision**: The paper reports errors ~3e-10 for N→∞. Our dense grid (N=400,000) achieves errors ~2e-5 due to grid discretization. The theoretical structural claims (contact sets, monotonicity, zero-premium for all-negative skews) are all confirmed at machine precision or well within stated thresholds.

## Artifacts
- `exp_support_resistance.csv` / `.png` — Exp 1 table and plots
- `exp_two_support.csv` / `.png` — Exp 2 table and plots
- `exp_other_regimes.csv` / `.png` — Exp 3 table and plots
- `exp_monotonicity.csv` / `.png` — Exp 4 table and plots
