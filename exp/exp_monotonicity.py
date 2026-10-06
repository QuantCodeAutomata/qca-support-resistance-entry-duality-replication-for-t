"""
Experiment 4 — Monotonicity of w(β) in the single-skew model.

Single skew level z = L = 1 with β ranging over {0.05, 0.10, …, 0.95}.
For each β we:
  1. Build the piecewise r-harmonic functions.
  2. Solve the exit problem and extract the lower boundary ā(β).
  3. Evaluate the analytical formula
         G₁(m) = κ₁ · m^{1-α₁} + κ₂ · m^{1-α₂}
     at m = ā(β) to obtain w(β) = G₁(ā(β)) - 1.
  4. Verify consistency: w(β) ≈ v(1) - 1  (direct numerical read-off).
  5. Verify strict monotonicity: w(β_{i+1}) > w(β_i) for all i.
  6. Print a summary table and save CSV + plot to results/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.harmonic import build_harmonic_functions, eval_at_point
from src.exit_problem import solve_exit, extract_exit_band
from src.roots_and_crossing import solve_roots, compute_kappas

# ---------------------------------------------------------------------------
# Global parameters
# ---------------------------------------------------------------------------
R, MU, SIGMA, L = 0.05, 0.01, 0.2, 1.0
X_GRID = np.geomspace(1e-4, 1e4, 400_000)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

BETAS = np.round(np.arange(0.05, 1.0, 0.05), 10)  # 0.05, 0.10, …, 0.95


def G1(m: float, kappa_1: float, kappa_2: float,
        alpha_1: float, alpha_2: float) -> float:
    """Analytical formula G₁(m) = κ₁·m^{1-α₁} + κ₂·m^{1-α₂}."""
    return kappa_1 * m ** (1 - alpha_1) + kappa_2 * m ** (1 - alpha_2)


def main() -> None:
    # Roots and kappas are independent of β
    alpha_1, alpha_2 = solve_roots(R, MU, SIGMA)
    kappa_1, kappa_2 = compute_kappas(alpha_1, alpha_2)
    print(f'α₁={alpha_1:.6f}, α₂={alpha_2:.6f}, κ₁={kappa_1:.6f}, κ₂={kappa_2:.6f}')

    rows = []
    for beta in BETAS:
        harm = build_harmonic_functions(
            R, MU, SIGMA,
            levels=[L], betas=[beta],
            x_grid=X_GRID,
        )
        varphi_r     = harm['varphi_r']
        F            = harm['F']
        psi_coeffs   = harm['psi_coeffs']
        varphi_coeffs = harm['varphi_coeffs']

        exit_res = solve_exit(X_GRID, g=X_GRID, varphi_r=varphi_r, F=F)
        v   = exit_res['v']
        K_g = exit_res['K_g']

        a_bar, _b = extract_exit_band(X_GRID, v, X_GRID)

        # G₁ formula evaluated at a_bar
        w_formula = G1(a_bar, kappa_1, kappa_2, alpha_1, alpha_2) - 1.0

        # Consistency: W(L) from piecewise evaluation
        psi_L = eval_at_point(L, alpha_1, alpha_2, psi_coeffs)
        phi_L = eval_at_point(L, alpha_1, alpha_2, varphi_coeffs)
        F_L   = psi_L / phi_L
        K_g_L = np.interp(F_L, F, K_g)
        W_1   = phi_L * K_g_L - L

        rows.append(dict(
            beta=beta,
            a_bar=a_bar,
            w_formula=w_formula,
            W_1_direct=W_1,
            consistency_err=abs(w_formula - W_1),
        ))

    df = pd.DataFrame(rows)

    # --- Verify strict monotonicity ------------------------------------------
    w_vals = df['w_formula'].values
    diffs  = np.diff(w_vals)
    monotone = bool(np.all(diffs > 0))
    print(f'\nStrict monotonicity of w(β): {monotone}')
    print(f'Min increment: {diffs.min():.4e}')
    print(f'w(0.05) = {w_vals[0]:.4e}   (expected ~ 6.2e-4)')
    print(f'w(0.95) = {w_vals[-1]:.4e}  (expected ~ 0.202)')
    print(f'Max consistency error |w_formula - W(1)|: {df["consistency_err"].max():.2e}')

    # --- Print table ---------------------------------------------------------
    pd.set_option('display.float_format', '{:.6f}'.format)
    print('\n', df.to_string(index=False))

    csv_path = os.path.join(RESULTS_DIR, 'exp_monotonicity.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nTable saved to {csv_path}')

    # --- Plot ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        'Experiment 4: Monotonicity of $w(\\beta)$, single-skew $z=1$',
        fontsize=13,
    )

    ax = axes[0]
    ax.plot(df['beta'], df['w_formula'],   'o-',  label=r'$w(\beta)=G_1(\bar a(\beta))-1$', lw=1.8)
    ax.plot(df['beta'], df['W_1_direct'],  's--', label=r'$v(1)-1$ (direct)', ms=4, lw=1.0)
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$w(\beta)$')
    ax.set_title('Entry premium $w(\\beta)$ vs skew coefficient')
    ax.legend()

    ax2 = axes[1]
    ax2.plot(df['beta'], df['a_bar'], 'o-', color='darkorange', lw=1.8)
    ax2.set_xlabel(r'$\beta$')
    ax2.set_ylabel(r'$\bar a(\beta)$')
    ax2.set_title('Lower exit boundary $\\bar a(\\beta)$ vs skew coefficient')
    ax2.axhline(L, color='grey', ls='--', lw=0.8, label='z=L=1')
    ax2.legend()

    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, 'exp_monotonicity.png')
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f'Plot saved to {png_path}')


if __name__ == '__main__':
    main()
