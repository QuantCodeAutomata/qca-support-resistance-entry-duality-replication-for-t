"""
Experiment 3 — Other sign regimes for (beta_L, beta_H).

Case A: Both betas negative (beta_L < 0, beta_H < 0).
  P+ = {} (empty).  The exit premium W = v - x is ≈ 0 everywhere, and the
  entry value u = 0.

Case B: Lower level negative, upper level positive (beta_L < 0, beta_H > 0).
  P+ = {H}.  Entry value equals the hitting-time formula
  u(x) = W(H) · E_x[e^{-r τ_H}].

For each scenario we:
  1. Build piecewise r-harmonic functions.
  2. Solve the exit problem and extract the exit band (a, b).
  3. Compute W = v - x; for Case A verify |W| < 2e-12.
  4. For Case B compute W(H) precisely, build theorem entry with P+ = {H},
     and compare to the hitting-time formula and brute-force on [0.3, 6].
  5. Print a summary table (reproducing Table 3) and save CSV + plot.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.harmonic import build_harmonic_functions, eval_at_point
from src.exit_problem import solve_exit, extract_exit_band
from src.entry_problem import (
    solve_entry_bruteforce,
    solve_entry_theorem,
    compute_hitting_time_formula,
)

# ---------------------------------------------------------------------------
# Global parameters
# ---------------------------------------------------------------------------
R, MU, SIGMA, L = 0.05, 0.01, 0.2, 1.0
X_GRID = np.geomspace(1e-4, 1e4, 400_000)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

PLOT_MASK = (X_GRID >= 0.3) & (X_GRID <= 6.0)

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
CASE_A = [
    dict(label='A1', H=1.5, beta_L=-0.5, beta_H=-0.5),
    dict(label='A2', H=1.3, beta_L=-0.9, beta_H=-0.3),
]

CASE_B = [
    dict(label='B1', H=3.0, beta_L=-0.5, beta_H=0.9,
         a_exp=1.876, b_exp=4.396, WH_exp=0.548912),
    dict(label='B2', H=1.2, beta_L=-0.2, beta_H=0.9,
         a_exp=0.889, b_exp=1.750, WH_exp=0.213649),
    dict(label='B3', H=1.2, beta_L=-0.6, beta_H=0.9,
         a_exp=1.000, b_exp=1.749, WH_exp=0.212365),
]


def _eval_at(pt: float, harm: dict, K_g: np.ndarray) -> tuple:
    """Return (psi, phi, F_pt, W_pt) using exact piecewise coefficients."""
    a1 = harm['alpha_1']
    a2 = harm['alpha_2']
    psi_val = eval_at_point(pt, a1, a2, harm['psi_coeffs'])
    phi_val = eval_at_point(pt, a1, a2, harm['varphi_coeffs'])
    F_pt    = psi_val / phi_val
    K_g_pt  = np.interp(F_pt, harm['F'], K_g)
    W_pt    = phi_val * K_g_pt - pt
    return psi_val, phi_val, F_pt, W_pt


def run_case_a(sc: dict) -> dict:
    """Case A: both betas negative, W ≈ 0, u = 0."""
    H, beta_L, beta_H = sc['H'], sc['beta_L'], sc['beta_H']
    harm = build_harmonic_functions(
        R, MU, SIGMA, levels=[L, H], betas=[beta_L, beta_H], x_grid=X_GRID,
    )
    varphi_r = harm['varphi_r']
    F        = harm['F']

    exit_res = solve_exit(X_GRID, g=X_GRID, varphi_r=varphi_r, F=F)
    v   = exit_res['v']
    K_g = exit_res['K_g']
    a, b = extract_exit_band(X_GRID, v, X_GRID)
    W = v - X_GRID

    # W should be effectively zero everywhere
    mask_inner = (X_GRID >= 0.1) & (X_GRID <= 10.0)
    max_W = float(np.max(np.abs(W[mask_inner])))

    return dict(
        label=sc['label'], H=H, beta_L=beta_L, beta_H=beta_H,
        a=a, b=b, max_W=max_W,
        W=W, u=np.zeros_like(W),
    )


def run_case_b(sc: dict) -> dict:
    """Case B: beta_L < 0, beta_H > 0 → P+ = {H}."""
    H, beta_L, beta_H = sc['H'], sc['beta_L'], sc['beta_H']
    harm = build_harmonic_functions(
        R, MU, SIGMA, levels=[L, H], betas=[beta_L, beta_H], x_grid=X_GRID,
    )
    varphi_r = harm['varphi_r']
    psi_r    = harm['psi_r']
    F        = harm['F']

    exit_res = solve_exit(X_GRID, g=X_GRID, varphi_r=varphi_r, F=F)
    v   = exit_res['v']
    K_g = exit_res['K_g']
    a, b = extract_exit_band(X_GRID, v, X_GRID)
    W = v - X_GRID

    # --- Precise evaluation at H -------------------------------------------
    psi_H, phi_H, F_H, W_H = _eval_at(H, harm, K_g)

    # --- Theorem entry: P+ = {H} -------------------------------------------
    W_support   = np.array([W_H])
    phi_support = np.array([phi_H])
    F_support   = np.array([F_H])
    u_theory = solve_entry_theorem(
        X_GRID, W_support, phi_support, F_support, varphi_r, F
    )

    # --- Hitting-time formula ------------------------------------------------
    u_hit = compute_hitting_time_formula(
        X_GRID, H, W_H, psi_r, psi_H, varphi_r, phi_H
    )

    # --- Brute-force entry --------------------------------------------------
    u_bf = solve_entry_bruteforce(X_GRID, W, varphi_r, F)['u']

    # --- Comparisons on [0.3, 6] -------------------------------------------
    max_err_hit_bf     = float(np.max(np.abs(u_hit[PLOT_MASK] - u_bf[PLOT_MASK])))
    max_err_theory_bf  = float(np.max(np.abs(u_theory[PLOT_MASK] - u_bf[PLOT_MASK])))

    return dict(
        label=sc['label'], H=H, beta_L=beta_L, beta_H=beta_H,
        a=a, b=b, W_H=W_H,
        a_exp=sc['a_exp'], b_exp=sc['b_exp'], WH_exp=sc['WH_exp'],
        max_err_hit_bf=max_err_hit_bf,
        max_err_theory_bf=max_err_theory_bf,
        W=W, u_hit=u_hit, u_theory=u_theory, u_bf=u_bf,
    )


def main() -> None:
    # --- Run all scenarios ---------------------------------------------------
    case_a_results = [run_case_a(sc) for sc in CASE_A]
    case_b_results = [run_case_b(sc) for sc in CASE_B]

    # --- Print / build table -------------------------------------------------
    rows = []

    print('=== Case A: both betas negative ===')
    for r in case_a_results:
        rows.append({
            'label':   r['label'],
            'H':       r['H'],
            'beta_L':  r['beta_L'],
            'beta_H':  r['beta_H'],
            'case':    'A',
            'a':       round(r['a'], 4) if not np.isnan(r['a']) else 'nan',
            'b':       round(r['b'], 4) if not np.isnan(r['b']) else 'nan',
            'W(H)':    'n/a',
            'max_W':   f"{r['max_W']:.2e}",
            'contact': '{}',
            'err_hit_bf':    'n/a',
            'err_theory_bf': 'n/a',
        })
        print(
            f"  {r['label']}: H={r['H']}, beta_L={r['beta_L']}, beta_H={r['beta_H']} → "
            f"max|W|={r['max_W']:.2e} (expect < 2e-12)"
        )

    print('\n=== Case B: beta_L < 0, beta_H > 0, P+ = {H} ===')
    for r in case_b_results:
        rows.append({
            'label':   r['label'],
            'H':       r['H'],
            'beta_L':  r['beta_L'],
            'beta_H':  r['beta_H'],
            'case':    'B',
            'a':       round(r['a'], 4),
            'b':       round(r['b'], 4),
            'W(H)':    round(r['W_H'], 6),
            'max_W':   'n/a',
            'contact': '{H}',
            'err_hit_bf':    f"{r['max_err_hit_bf']:.2e}",
            'err_theory_bf': f"{r['max_err_theory_bf']:.2e}",
        })
        print(
            f"  {r['label']}: H={r['H']}, beta_L={r['beta_L']}, beta_H={r['beta_H']} → "
            f"a={r['a']:.4f} (exp {r['a_exp']}), b={r['b']:.4f} (exp {r['b_exp']}), "
            f"W(H)={r['W_H']:.6f} (exp {r['WH_exp']}), "
            f"err(hit,bf)={r['max_err_hit_bf']:.2e}, "
            f"err(theory,bf)={r['max_err_theory_bf']:.2e}"
        )

    df = pd.DataFrame(rows)
    print('\n', df.to_string(index=False))
    csv_path = os.path.join(RESULTS_DIR, 'exp_other_regimes.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nTable saved to {csv_path}')

    # --- Plot ----------------------------------------------------------------
    n_b = len(case_b_results)
    n_a = len(case_a_results)
    n_total = n_a + n_b
    ncols = 3
    nrows = (n_total + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    fig.suptitle('Experiment 3: Other Sign Regimes', fontsize=13)
    axes_flat = axes.flatten() if nrows > 1 else np.array(axes).flatten()
    x_plot = X_GRID[PLOT_MASK]

    for idx, r in enumerate(case_a_results):
        ax = axes_flat[idx]
        ax.plot(x_plot, r['W'][PLOT_MASK], label=r'$W(x)$', lw=1.6)
        ax.set_title(
            f"Case A: {r['label']}, H={r['H']}, "
            f"$\\beta_L$={r['beta_L']}, $\\beta_H$={r['beta_H']}\n"
            f"max|W|={r['max_W']:.1e}",
            fontsize=8,
        )
        ax.set_xlabel('x')
        ax.legend(fontsize=7)
        ax.axhline(0, color='black', lw=0.5)

    for jdx, r in enumerate(case_b_results):
        ax = axes_flat[n_a + jdx]
        ax.plot(x_plot, r['W'][PLOT_MASK],       label=r'$W(x)$',        lw=1.8)
        ax.plot(x_plot, r['u_bf'][PLOT_MASK],    label=r'$u$ brute-force', lw=1.4, ls='--')
        ax.plot(x_plot, r['u_hit'][PLOT_MASK],   label=r'$u$ hit-time',   lw=1.0, ls=':', color='green')
        ax.axvline(r['H'], color='brown', ls=':', lw=0.8)
        ax.set_title(
            f"Case B: {r['label']}, H={r['H']}, "
            f"$\\beta_L$={r['beta_L']}, $\\beta_H$={r['beta_H']}\n"
            f"contact={{H}}, err_hit={r['max_err_hit_bf']:.1e}",
            fontsize=8,
        )
        ax.set_xlabel('x')
        ax.legend(fontsize=7)

    # Hide any unused axes
    for k in range(n_a + n_b, len(axes_flat)):
        axes_flat[k].set_visible(False)

    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, 'exp_other_regimes.png')
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f'Plot saved to {png_path}')


if __name__ == '__main__':
    main()
