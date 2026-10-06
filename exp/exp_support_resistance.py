"""
Experiment 1 — Support-resistance regime (beta_L > 0 > beta_H).

Three scenarios with L=1 acting as support (beta_L > 0) and H acting as
resistance (beta_H < 0).  For each scenario we:
  1. Build piecewise r-harmonic functions on the geometric grid.
  2. Solve the exit (optimal-selling) problem.
  3. Extract the exit band (a, b).
  4. Compute the entry (optimal-buying) value W = v - x and u = entry value.
  5. Use the theorem-based formula (P+ = {L}) and compare to brute-force.
  6. Print a summary table and save CSV + plot to results/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.harmonic import build_harmonic_functions, eval_at_point
from src.exit_problem import solve_exit, extract_exit_band
from src.entry_problem import solve_entry_bruteforce, solve_entry_theorem

# ---------------------------------------------------------------------------
# Global parameters
# ---------------------------------------------------------------------------
R, MU, SIGMA, L = 0.05, 0.01, 0.2, 1.0
X_GRID = np.geomspace(1e-4, 1e4, 400_000)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Scenarios: name, H, beta_L, beta_H, expected a, expected b, expected W(L)
SCENARIOS = [
    dict(name='A', H=1.60, beta_L=0.9, beta_H=-0.30,
         a_exp=0.6253, b_exp=1.4654, WL_exp=0.182971),
    dict(name='B', H=1.15, beta_L=0.9, beta_H=-0.20,
         a_exp=0.6683, b_exp=1.3422, WL_exp=0.137656),
    dict(name='C', H=1.15, beta_L=0.9, beta_H=-0.50,
         a_exp=0.7003, b_exp=1.1500, WL_exp=0.109221),
]

# Comparison window
PLOT_MASK = (X_GRID >= 0.3) & (X_GRID <= 3.0)


def run_scenario(sc: dict) -> dict:
    """Run one support-resistance scenario and return result dict."""
    H, beta_L, beta_H = sc['H'], sc['beta_L'], sc['beta_H']

    # --- Harmonic functions ---------------------------------------------------
    harm = build_harmonic_functions(
        R, MU, SIGMA,
        levels=[L, H], betas=[beta_L, beta_H],
        x_grid=X_GRID,
    )
    alpha_1      = harm['alpha_1']
    alpha_2      = harm['alpha_2']
    psi_r        = harm['psi_r']
    varphi_r     = harm['varphi_r']
    F            = harm['F']
    psi_coeffs   = harm['psi_coeffs']
    varphi_coeffs = harm['varphi_coeffs']

    # --- Exit problem --------------------------------------------------------
    exit_res = solve_exit(X_GRID, g=X_GRID, varphi_r=varphi_r, F=F)
    v   = exit_res['v']
    K_g = exit_res['K_g']
    a, b = extract_exit_band(X_GRID, v, X_GRID)

    # --- Gross entry reward W = v - x ----------------------------------------
    W = v - X_GRID

    # --- Precise evaluation of W(L) via piecewise coefficients ---------------
    psi_L  = eval_at_point(L, alpha_1, alpha_2, psi_coeffs)
    phi_L  = eval_at_point(L, alpha_1, alpha_2, varphi_coeffs)
    F_L    = psi_L / phi_L
    K_g_L  = np.interp(F_L, F, K_g)
    v_L    = phi_L * K_g_L
    W_L    = v_L - L

    # --- Theorem-based entry: P+ = {L} (only L has beta_L > 0) ---------------
    W_support       = np.array([W_L])
    phi_support     = np.array([phi_L])
    F_support       = np.array([F_L])
    u_theory = solve_entry_theorem(
        X_GRID, W_support, phi_support, F_support, varphi_r, F
    )

    # --- Brute-force entry ----------------------------------------------------
    u_bf = solve_entry_bruteforce(X_GRID, W, varphi_r, F)['u']

    # --- Comparison on [0.3, 3] -----------------------------------------------
    max_err = float(np.max(np.abs(u_theory[PLOT_MASK] - u_bf[PLOT_MASK])))

    return dict(
        name=sc['name'], H=H, beta_L=beta_L, beta_H=beta_H,
        a=a, b=b, W_L=W_L,
        a_exp=sc['a_exp'], b_exp=sc['b_exp'], WL_exp=sc['WL_exp'],
        max_err=max_err,
        # arrays for plotting
        W=W, u_theory=u_theory, u_bf=u_bf,
    )


def main() -> None:
    results = [run_scenario(sc) for sc in SCENARIOS]

    # --- Summary table -------------------------------------------------------
    rows = []
    for r in results:
        rows.append({
            'Scenario':  r['name'],
            'H':         r['H'],
            'beta_L':    r['beta_L'],
            'beta_H':    r['beta_H'],
            'a':         round(r['a'],    4),
            'b':         round(r['b'],    4),
            'a_exp':     r['a_exp'],
            'b_exp':     r['b_exp'],
            'W(L)':      round(r['W_L'],  6),
            'W(L)_exp':  r['WL_exp'],
            'max_err':   f"{r['max_err']:.2e}",
        })
        print(
            f"Scenario {r['name']}: "
            f"a={r['a']:.4f} (exp {r['a_exp']}), "
            f"b={r['b']:.4f} (exp {r['b_exp']}), "
            f"W(L)={r['W_L']:.6f} (exp {r['WL_exp']}), "
            f"max|u_theory-u_bf|={r['max_err']:.2e}"
        )

    df = pd.DataFrame(rows)
    print('\n', df.to_string(index=False))
    csv_path = os.path.join(RESULTS_DIR, 'exp_support_resistance.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nTable saved to {csv_path}')

    # --- Plot ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Experiment 1: Support-Resistance Regime ($\\beta_L>0>\\beta_H$)')

    x_plot = X_GRID[PLOT_MASK]
    for ax, r in zip(axes, results):
        ax.plot(x_plot, r['W'][PLOT_MASK],        label=r'$W(x)=v(x)-x$', linewidth=1.8)
        ax.plot(x_plot, r['u_bf'][PLOT_MASK],     label=r'$u(x)$ brute-force',
                linewidth=1.4, linestyle='--')
        ax.plot(x_plot, r['u_theory'][PLOT_MASK], label=r'$u(x)$ theorem',
                linewidth=1.0, linestyle=':', color='red')
        ax.axvline(L, color='grey', linestyle=':', linewidth=0.8, label='L=1')
        ax.set_title(
            f"Scenario {r['name']}: H={r['H']}, "
            f"$\\beta_L$={r['beta_L']}, $\\beta_H$={r['beta_H']}"
        )
        ax.set_xlabel('x')
        ax.set_ylabel('Value')
        ax.legend(fontsize=7)
        ax.set_xlim(0.3, 3.0)

    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, 'exp_support_resistance.png')
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f'Plot saved to {png_path}')


if __name__ == '__main__':
    main()
