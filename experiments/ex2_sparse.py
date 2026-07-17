#!/usr/bin/env python3
"""Example 4.2 — Sparse tensor systems from SuiteSparse.

Matrices: nos5, ash85, Cities, WorldCities
Downloaded on demand via ssgetpy.
"""
import numpy as np
import torch
import ssgetpy
import matplotlib.pyplot as plt
from scipy.io import mmread

from utils import (device, dtype, SEED, OUTDIR, SS_DATA, METHODS, METHOD_STYLE,
                   sparse_mtx_to_tensor, tgdbek_with_z_rse, save_fig,
                   trek_algorithm, trebk_algorithm, tregbk_algorithm, make_partitions)
import os


def main():
    print("\n" + "*"*65)
    print("  EXAMPLE 4.2  Sparse tensor systems (SuiteSparse)")
    print("*"*65)

    from tensor_toolbox.tensorLinalg import (t_product as tp, t_pinv_apply as tpa,
                                              t_frobenius_norm as tfn)

    # (name, N3=p=K, noise, max_iter, eta, rcond_tgdbek)
    configs = [
        ("nos5",        12, 1e-2,  300, 0.5, 1e-1),
        ("ash85",        5, 1e-3,  300, 0.5, 1e-3),
        ("Cities",       2, 1e-3, 6000, 0.5, 1e-2),
        ("WorldCities",  5, 1e-1, 6000, 0.7, 1e-2),
        ("gre_216a",     9, 1e-3, 6000, 0.7, 1e-2),
    ]
    DELTA = 0.7
    TAU   = 10
    TOL   = 1e-6

    all_rows = []

    for (name, N3, noise, T, eta, rcond_tgdbek) in configs:
        ssgetpy.fetch(name, location=SS_DATA)
        path = os.path.join(SS_DATA, name, f"{name}.mtx")
        M_sp = mmread(path)
        N2   = M_sp.shape[1] // N3

        print(f"\n  Matrix: {name}  N2={N2}, N3={N3}, noise={noise}, T={T}")
        A = sparse_mtx_to_tensor(name, N2, N3)
        m, n, p = A.shape
        K = N3
        print(f"    Tensor shape: {tuple(A.shape)}, K={K}")

        torch.manual_seed(1)
        Xs_t    = torch.randn(n, K, p, device=device, dtype=dtype)
        B_clean = tp(A, Xs_t)
        E       = torch.randn_like(B_clean)
        E       = E / (tfn(E) + 1e-12)
        B_t     = B_clean + noise * tfn(B_clean) * E
        Xls_t   = tpa(A, B_t, rcond=1e-3)

        row_parts = make_partitions(m, tau=TAU, sequential=True)
        col_parts = make_partitions(n, tau=TAU, sequential=True)
        hists = {}
        time_hists = {}

        for method, fn, kw in [
            ("TREK",   trek_algorithm,   dict(tol=TOL, seed=SEED)),
            ("TREBK",  trebk_algorithm,  dict(row_partitions=row_parts,
                                              col_partitions=col_parts, tol=TOL, seed=SEED)),
            ("TREGBK", tregbk_algorithm, dict(delta=DELTA, row_partitions=row_parts,
                                              tol=TOL, seed=SEED)),
        ]:
            (_, k, h, th), t = fn(A, B_t, T, Xls_t, **kw)
            rse = float(h[-1])
            hists[method] = h
            time_hists[method] = th
            print(f"    {method:<8}  IT={k:5}  CPU={t:.3f}s  RSE={rse:.3e}")
            all_rows.append((name, method, k, t, rse))

        (_, k, h_tg, h_z_tg, th_tg), t = tgdbek_with_z_rse(A, B_t, T, Xls_t,
                                                             eta=eta, tol=TOL,
                                                             rcond=rcond_tgdbek)
        rse = float(h_tg[-1])
        hists["TGDBEK"] = h_tg; hists["TGDBEK_Z"] = h_z_tg
        time_hists["TGDBEK"] = th_tg
        print(f"    TGDBEK   IT={k:5}  CPU={t:.3f}s  RSE={rse:.3e}")
        all_rows.append((name, "TGDBEK", k, t, rse))

        # Convergence plot for this matrix
        fig, (ax_x, ax_r) = plt.subplots(1, 2, figsize=(12, 5))
        for method in METHODS:
            h = hists.get(method)
            if h is None:
                continue
            st = METHOD_STYLE[method]
            h_plot = np.where(np.isfinite(h), h, np.nan)
            ax_x.semilogy(h_plot, label=method, color=st["color"],
                          linestyle=st["linestyle"], linewidth=1.5,
                          markevery=max(1, len(h_plot)//15), marker=st["marker"],
                          markersize=4)
        ax_x.axhline(TOL, color="k", linestyle=":", linewidth=1, label=f"tol={TOL:.0e}")
        ax_x.set_xlabel("Iteration", fontsize=12)
        ax_x.set_ylabel(r"$\mathrm{RSE}^X$", fontsize=12)
        ax_x.set_title(f"$\\mathrm{{RSE}}^X$ vs Iteration — {name}", fontsize=11)
        ax_x.legend(fontsize=8); ax_x.grid(True, alpha=0.3)

        if name == "nos5":
            for method in METHODS:
                h = hists.get(method)
                th = time_hists.get(method)
                if h is None or th is None:
                    continue
                st = METHOD_STYLE[method]
                h_plot = np.where(np.isfinite(h), h, np.nan)
                ax_r.semilogy(th, h_plot, label=method, color=st["color"],
                              linestyle=st["linestyle"], linewidth=1.5,
                              markevery=max(1, len(h_plot)//15), marker=st["marker"],
                              markersize=4)
            ax_r.axhline(TOL, color="k", linestyle=":", linewidth=1, label=f"tol={TOL:.0e}")
            ax_r.set_xlabel("CPU time (s)", fontsize=12)
            ax_r.set_ylabel(r"$\mathrm{RSE}^X$", fontsize=12)
            ax_r.set_title(f"$\\mathrm{{RSE}}^X$ vs CPU time — {name}", fontsize=11)
            ax_r.legend(fontsize=8); ax_r.grid(True, alpha=0.3)
        else:
            ax_r.semilogy(hists["TGDBEK_Z"], color=METHOD_STYLE["TGDBEK"]["color"],
                          linestyle="--", linewidth=2.0, label="TGDBEK $\\mathrm{RSE}^Z$")
            ax_r.axhline(TOL, color="k", linestyle=":", linewidth=1, label=f"tol={TOL:.0e}")
            ax_r.set_xlabel("Iteration", fontsize=12)
            ax_r.set_ylabel(r"$\mathrm{RSE}^Z$", fontsize=12)
            ax_r.set_title(f"$\\mathrm{{RSE}}^Z$ (TGDBEK) — {name}", fontsize=11)
            ax_r.legend(fontsize=9); ax_r.grid(True, alpha=0.3)

        plt.tight_layout()
        save_fig(fig, f"ex2-{name.lower()}-convergence")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("  TABLE 2  Sparse tensor systems")
    print(f"  {'Matrix':<15} {'Method':<10} {'IT':>7} {'CPU (s)':>9} {'RSE':>13}")
    print(f"  {'-'*57}")
    cur_mat = None
    for (mat, meth, it, cpu, rse) in all_rows:
        if mat != cur_mat:
            if cur_mat is not None:
                print(f"  {'-'*57}")
            cur_mat = mat
        rse_str = f"{rse:.3e}" if rse == rse else "NaN"
        bold = " *" if meth == "TGDBEK" else "  "
        print(f"{bold} {mat:<15} {meth:<10} {it:>7} {cpu:>9.3f} {rse_str:>13}")
    print(f"{'='*68}\n")

    print("  Example 4.2 complete.\n")
    print(f"  Figures saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
