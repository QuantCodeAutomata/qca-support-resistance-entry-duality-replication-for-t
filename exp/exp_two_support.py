"""
Experiment 2 — Two-support regime (beta_L > 0, beta_H > 0).

Seven scenarios where both levels L and H have positive skew coefficients,
so P+ = {L, H}.  For each scenario we:
  1. Build piecewise r-harmonic functions.
  2. Solve the exit problem and extract the exit band (a, b).
  3. Evaluate W(L) and W(H) precisely via piecewise coefficients.
  4. Compute the ratio-window classification (rho, underline_rho, overline_rho).
  5. Classify the entry contact set from the finite upper hull.
  6. Build theorem-based and brute-force entry values; compare them.
  7. Print a summary table and save CSV + plot to results/.
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
    classify_entry_contact,
    compute_ratio_window,
)

# ---------------------------------------------------------------------------
# Global parameters
# ---------------------------------------------------------------------------
R, MU, SIGMA, L = 0.05, 0.01, 0.2, 1.0
X_GRID = np.geomspace(1e-4, 1e4, 400_000)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Scenarios: H, beta_L, beta_H, expected contact label, expected rho bounds
SCENARIOS = [
    dict(H=3.0, beta_L=0.9, beta_H=0.9,
         contact_exp='{L,H}', rho_exp=0.3333, rho_lo=0.2794, rho_hi=36.91),
    dict(H=3.0, beta_L=0.5, beta_H=0.9,
         contact_exp='{H}',   rho_exp=0.1098, rho_lo=None,   rho_hi=0.2091),
    dict(H=3.0, beta_L=0.9, beta_H=0.5,
         contact_exp='{L,H}', rho_exp=1.0118, rho_lo=0.2794, rho_hi=8.022),
    dict(H=1.5, beta_L=0.9, beta_H=0.9,
         contact_exp='{L,H}', rho_exp=1.3964, rho_lo=0.7845, rho_hi=11.28),
    dict(H=1.5, beta_L=0.6, beta_H=0.9,
         contact_exp='{L,H}', rho_exp=0.8757, rho_lo=0.6895, rho_hi=11.28),
    dict(H=1.3, beta_L=0.9, beta_H=0.3,
         contact_exp='{L}',   rho_exp=2.5287, rho_lo=0.8934, rho_hi=1.718),
    dict(H=2.2, beta_L=0.7, beta_H=0.8,
         contact_exp='{H}',   rho_exp=0.3540, rho_lo=0.4135, rho_hi=11.91),
]

PLOT_MASK = (X_GRID >= 0.3) & (X_GRID <= 6.0)


def _eval_at(pt: float, harm: dict) -> tuple:
    """Return (psi(pt), phi(pt), F(pt)) using exact piecewise coefficients."""
    a1 = harm['alpha_1']
    a2 = harm['alpha_2']
    psi_val = eval_at_point(pt, a1, a2, harm['psi_coeffs'])
    phi_val = eval_at_point(pt, a1, a2, harm['varphi_coeffs'])
    return psi_val, phi_val, psi_val / phi_val


def run_scenario(sc: dict) -> dict:
    H, beta_L, beta_H = sc['H'], sc['beta_L'], sc['beta_H']

    # --- Harmonic functions --------------------------------------------------
    harm = build_harmonic_functions(
        R, MU, SIGMA,
        levels=[L, H], betas=[beta_L, beta_H],
        x_grid=X_GRID,
    )
    varphi_r = harm['varphi_r']
    F        = harm['F']

    # --- Exit problem --------------------------------------------------------
    exit_res = solve_exit(X_GRID, g=X_GRID, varphi_r=varphi_r, F=F)
    v   = exit_res['v']
    K_g = exit_res['K_g']
    a, b = extract_exit_band(X_GRID, v, X_GRID)
    W = v - X_GRID

    # --- Precise W at L and H ------------------------------------------------
    psi_L, phi_L, F_L = _eval_at(L, harm)
    psi_H, phi_H, F_H = _eval_at(H, harm)

    K_g_L = np.interp(F_L, F, K_g)
    K_g_H = np.interp(F_H, F, K_g)
    W_L = phi_L * K_g_L - L
    W_H = phi_H * K_g_H - H

    # --- Ratio window --------------------------------------------------------
    rw = compute_ratio_window(psi_L, psi_H, phi_L, phi_H, W_L, W_H)
    rho         = rw['rho']
    under_rho   = rw['underline_rho']
    over_rho    = rw['overline_rho']
    predicted   = rw['predicted']

    # --- Entry contact set ---------------------------------------------------
    # classify_entry_contact uses the finite upper-hull of P+ points.
    # For rho > overline_rho the finite hull still includes H (rightmost point),
    # but the true contact (confirmed by brute-force) is {L} only.
    # We therefore use the ratio-window prediction to select the correct
    # support for the theorem formula.
    W_support_all   = np.array([W_L, W_H])
    phi_support_all = np.array([phi_L, phi_H])
    F_support_all   = np.array([F_L, F_H])

    gamma_hull = classify_entry_contact([L, H], W_support_all, phi_support_all, F_support_all)
    contact_hull_str = '{' + ','.join(str(g) for g in sorted(gamma_hull)) + '}'

    # Ratio-window-based contact (theoretically correct for two-level case)
    if predicted == 'H':
        gamma = [H]
        W_th   = np.array([W_H])
        phi_th = np.array([phi_H])
        F_th   = np.array([F_H])
    elif predicted == 'L':
        gamma = [L]
        W_th   = np.array([W_L])
        phi_th = np.array([phi_L])
        F_th   = np.array([F_L])
    else:  # 'L,H'
        gamma = [L, H]
        W_th   = W_support_all
        phi_th = phi_support_all
        F_th   = F_support_all
    contact_str = '{' + ','.join(str(g) for g in sorted(gamma)) + '}'

    # --- Theorem-based entry -------------------------------------------------
    u_theory = solve_entry_theorem(
        X_GRID, W_th, phi_th, F_th, varphi_r, F
    )

    # --- Brute-force entry ---------------------------------------------------
    u_bf = solve_entry_bruteforce(X_GRID, W, varphi_r, F)['u']

    # --- Comparison on plot window -------------------------------------------
    max_err = float(np.max(np.abs(u_theory[PLOT_MASK] - u_bf[PLOT_MASK])))

    return dict(
        H=H, beta_L=beta_L, beta_H=beta_H,
        a=a, b=b, W_L=W_L, W_H=W_H,
        rho=rho, under_rho=under_rho, over_rho=over_rho,
        predicted=predicted, contact=contact_str,
        contact_hull=contact_hull_str,
        contact_exp=sc['contact_exp'], rho_exp=sc['rho_exp'],
        max_err=max_err,
        W=W, u_theory=u_theory, u_bf=u_bf,
    )


def main() -> None:
    results = [run_scenario(sc) for sc in SCENARIOS]

    # --- Summary table -------------------------------------------------------
    rows = []
    for r in results:
        rows.append({
            'H':            r['H'],
            'beta_L':       r['beta_L'],
            'beta_H':       r['beta_H'],
            'a':            round(r['a'], 4),
            'b':            round(r['b'], 4),
            'W(L)':         round(r['W_L'], 6),
            'W(H)':         round(r['W_H'], 6),
            'rho':          round(r['rho'], 4),
            'rho_lo':       round(r['under_rho'], 4),
            'rho_hi':       round(r['over_rho'], 4),
            'predicted':    r['predicted'],
            'contact':      r['contact'],
            'contact_hull': r['contact_hull'],
            'contact_exp':  r['contact_exp'],
            'max_err':      f"{r['max_err']:.2e}",
        })
        print(
            f"H={r['H']:.1f} beta_L={r['beta_L']} beta_H={r['beta_H']}: "
            f"a={r['a']:.4f}, b={r['b']:.4f}, "
            f"rho={r['rho']:.4f} in [{r['under_rho']:.4f}, {r['over_rho']:.4f}], "
            f"contact={r['contact']} (hull={r['contact_hull']}, exp {r['contact_exp']}), "
            f"max_err={r['max_err']:.2e}"
        )

    df = pd.DataFrame(rows)
    print('\n', df.to_string(index=False))
    csv_path = os.path.join(RESULTS_DIR, 'exp_two_support.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nTable saved to {csv_path}')

    # --- Plot: 7 subplots in a 4x2 grid (last cell empty) -------------------
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(
        'Experiment 2: Two-Support Regime ($\\beta_L>0,\\,\\beta_H>0$)',
        fontsize=13,
    )
    axes_flat = axes.flatten()
    x_plot = X_GRID[PLOT_MASK]

    for idx, r in enumerate(results):
        ax = axes_flat[idx]
        ax.plot(x_plot, r['W'][PLOT_MASK],        label=r'$W(x)$',        lw=1.8)
        ax.plot(x_plot, r['u_bf'][PLOT_MASK],     label=r'$u$ brute-force', lw=1.4, ls='--')
        ax.plot(x_plot, r['u_theory'][PLOT_MASK], label=r'$u$ theorem',    lw=1.0, ls=':', color='red')
        ax.axvline(L,    color='grey',  ls=':', lw=0.8)
        ax.axvline(r['H'], color='brown', ls=':', lw=0.8)
        ax.set_title(
            f"H={r['H']}, $\\beta_L$={r['beta_L']}, $\\beta_H$={r['beta_H']}\n"
            f"contact={r['contact']}",
            fontsize=8,
        )
        ax.set_xlabel('x', fontsize=7)
        ax.legend(fontsize=6)
        ax.set_xlim(0.3, min(r['H'] * 2, 6.0))

    # Hide the unused last subplot
    axes_flat[-1].set_visible(False)
    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, 'exp_two_support.png')
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f'Plot saved to {png_path}')


if __name__ == '__main__':
    main()
