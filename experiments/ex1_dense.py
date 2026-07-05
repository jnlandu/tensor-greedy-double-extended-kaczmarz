#!/usr/bin/env python3
"""Example 4.1 — Dense overdetermined tensor systems.

A ∈ R^{500×n×10}, n ∈ {20,30,40,50,60,80}, noise=1e-3
5 independent trials, max_iter=800, eta=0.6, tau=10
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import (device, dtype, SEED, OUTDIR, METHODS, METHOD_STYLE,
                   tgdbek_with_z_rse, save_fig, tgdbek_algorithm_faithful,
                   trek_algorithm, trebk_algorithm, tregbk_algorithm, make_partitions)


def main():
    print("\n" + "█"*65)
    print("  EXAMPLE 4.1  Dense overdetermined tensor systems")
    print("  A ∈ R^{500×n×10}, n ∈ {20,30,40,50,60,80}, noise=1e-3")
    print("  5 independent trials, max_iter=800, eta=0.6, tau=10")
    print("█"*65)

    from tensor_toolbox.tensorLinalg import t_product as tp, t_pinv_apply as tpa

    N_vals   = [20, 30, 40, 50, 60, 80]
    M, P, K  = 500, 10, 10
    T        = 800
    ETA      = 0.6
    DELTA    = 0.7
    TAU      = 10
    NOISE    = 1e-3
    N_TRIALS = 5

    rows_it  = {m: [] for m in METHODS}
    rows_cpu = {m: [] for m in METHODS}
    rng = np.random.default_rng(SEED)

    for n in N_vals:
        print(f"\n  n = {n} ...")
        A_trials, B_trials, Xls_trials = [], [], []
        for trial in range(N_TRIALS):
            A_np  = rng.standard_normal((M, n, P)).astype(np.float32)
            Xs_np = rng.standard_normal((n, K, P)).astype(np.float32)
            A_t   = torch.from_numpy(A_np).to(device=device, dtype=dtype)
            Xs_t  = torch.from_numpy(Xs_np).to(device=device, dtype=dtype)
            B_clean = tp(A_t, Xs_t)
            g = torch.Generator(device=device); g.manual_seed(SEED + trial)
            zeta = torch.randn(B_clean.shape, device=device, dtype=dtype, generator=g)
            eps  = NOISE * torch.linalg.norm(B_clean) / (torch.linalg.norm(zeta) + 1e-12) * zeta
            B_t  = B_clean + eps
            Xls_t = tpa(A_t, B_t, rcond=1e-12)
            A_trials.append(A_t); B_trials.append(B_t); Xls_trials.append(Xls_t)

        row_parts = make_partitions(M, tau=TAU, sequential=True)
        col_parts = make_partitions(n, tau=TAU, sequential=True)
        accum = {m: dict(it=[], cpu=[]) for m in METHODS}

        for i in range(N_TRIALS):
            A_t, B_t, Xls_t = A_trials[i], B_trials[i], Xls_trials[i]

            (_, k, _, _), t = trek_algorithm(A_t, B_t, T, Xls_t, tol=1e-5, seed=SEED + i)
            accum["TREK"]["it"].append(k); accum["TREK"]["cpu"].append(t)

            (_, k, _, _), t = trebk_algorithm(A_t, B_t, T, Xls_t,
                row_partitions=row_parts, col_partitions=col_parts, tol=1e-5, seed=SEED + i)
            accum["TREBK"]["it"].append(k); accum["TREBK"]["cpu"].append(t)

            (_, k, _, _), t = tregbk_algorithm(A_t, B_t, T, Xls_t, delta=DELTA,
                row_partitions=row_parts, tol=1e-5, seed=SEED + i)
            accum["TREGBK"]["it"].append(k); accum["TREGBK"]["cpu"].append(t)

            (_, k, _, _), t = tgdbek_algorithm_faithful(A_t, B_t, T, Xls_t,
                eta=ETA, tol=1e-5, rcond=1e-12)
            accum["TGDBEK"]["it"].append(k); accum["TGDBEK"]["cpu"].append(t)

        for m in METHODS:
            rows_it[m].append(round(np.mean(accum[m]["it"])))
            rows_cpu[m].append(round(np.mean(accum[m]["cpu"]), 3))
            print(f"    {m:<8}  IT={rows_it[m][-1]:4}  CPU={rows_cpu[m][-1]:.3f}s")

    # ── Table ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("  TABLE 1  IT and CPU (averaged over 5 trials)")
    print(f"  {'Method':<8}", end="")
    for n in N_vals:
        print(f"  {'n='+str(n):>12}", end="")
    print(); print(f"  {'-'*74}")
    for m in METHODS:
        print(f"  {m:<8}", end="")
        for i in range(len(N_vals)):
            print(f"  {rows_it[m][i]:>5} / {rows_cpu[m][i]:.3f}", end="")
        print()
    print(f"{'='*78}\n")

    # ── IT vs n and CPU vs n ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for m in METHODS:
        st = METHOD_STYLE[m]
        axes[0].plot(N_vals, rows_it[m],  label=m, linewidth=1.8, **st)
        axes[1].plot(N_vals, rows_cpu[m], label=m, linewidth=1.8, **st)
    for ax, ylabel, title in zip(axes,
                                  ["Iterations (IT)", "CPU time (s)"],
                                  ["Iteration count vs n", "CPU time vs n"]):
        ax.set_xlabel("n (column dimension)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); save_fig(fig, "ex1-IT-CPU", "ex1")

    for col, key, stem in [("Iterations (IT)", rows_it, "ex1-IT"),
                            ("CPU time (s)",    rows_cpu, "ex1-CPU")]:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        for m in METHODS:
            st = METHOD_STYLE[m]
            ax2.plot(N_vals, key[m], label=m, linewidth=1.8, **st)
        ax2.set_xlabel("n", fontsize=12); ax2.set_ylabel(col, fontsize=12)
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
        plt.tight_layout(); save_fig(fig2, stem)

    # ── RSE convergence for n=20 and n=80 ────────────────────────────────────
    # Collect histories (RSE and wall-clock time) for both iteration and CPU plots
    _conv_data = {}   # {n_plot: {method: {"h": array, "t": array, "hz": array, "tz": array}}}
    for n_plot in [20, 80]:
        rng2 = np.random.default_rng(SEED)
        A_np  = rng2.standard_normal((M, n_plot, P)).astype(np.float32)
        Xs_np = rng2.standard_normal((n_plot, K, P)).astype(np.float32)
        A_t   = torch.from_numpy(A_np).to(device=device, dtype=dtype)
        Xs_t  = torch.from_numpy(Xs_np).to(device=device, dtype=dtype)
        B_clean = tp(A_t, Xs_t)
        g = torch.Generator(device=device); g.manual_seed(SEED)
        zeta = torch.randn(B_clean.shape, device=device, dtype=dtype, generator=g)
        eps  = NOISE * torch.linalg.norm(B_clean) / (torch.linalg.norm(zeta) + 1e-12) * zeta
        B_t  = B_clean + eps
        Xls_t = tpa(A_t, B_t, rcond=1e-12)
        row_p = make_partitions(M,      tau=TAU, sequential=True)
        col_p = make_partitions(n_plot, tau=TAU, sequential=True)

        nd = {}
        for method, fn, kw in [
            ("TREK",   trek_algorithm,   dict(tol=1e-5, seed=SEED)),
            ("TREBK",  trebk_algorithm,  dict(row_partitions=row_p, col_partitions=col_p,
                                              tol=1e-5, seed=SEED)),
            ("TREGBK", tregbk_algorithm, dict(delta=DELTA, row_partitions=row_p,
                                              tol=1e-5, seed=SEED)),
        ]:
            (_, _, h, th), _ = fn(A_t, B_t, T, Xls_t, **kw)
            nd[method] = dict(h=h, t=th)

        (_, _, h_tg, h_z, th_tg), _ = tgdbek_with_z_rse(A_t, B_t, T, Xls_t,
                                                           eta=ETA, tol=1e-5, rcond=1e-12)
        nd["TGDBEK"] = dict(h=h_tg, t=th_tg, hz=h_z)
        _conv_data[n_plot] = nd

    # Fig: RSE^X vs iteration
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
    for ax3, n_plot in zip(axes3, [20, 80]):
        nd = _conv_data[n_plot]
        for method in METHODS:
            st = METHOD_STYLE[method]
            ax3.semilogy(nd[method]["h"], label=method, color=st["color"],
                         linestyle=st["linestyle"], linewidth=1.5)
        ax3.axhline(1e-5, color="k", linestyle=":", linewidth=1)
        ax3.set_xlabel("Iteration", fontsize=11); ax3.set_ylabel("RSE", fontsize=11)
        ax3.set_title(f"n = {n_plot}", fontsize=11)
        ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)
    plt.tight_layout(); save_fig(fig3, "ex1-rse-it")

    # Fig: RSE vs CPU time (X-RSE all methods + TGDBEK Z-RSE)
    fig_cpu, axes_cpu = plt.subplots(1, 2, figsize=(12, 5))
    for ax_c, n_plot in zip(axes_cpu, [20, 80]):
        nd = _conv_data[n_plot]
        for method in METHODS:
            st = METHOD_STYLE[method]
            ax_c.semilogy(nd[method]["t"], nd[method]["h"],
                          label=method, color=st["color"],
                          linestyle=st["linestyle"], linewidth=1.5)
        ax_c.axhline(1e-5, color="k", linestyle=":", linewidth=1)
        ax_c.set_xlabel("CPU time (s)", fontsize=11); ax_c.set_ylabel("RSE", fontsize=11)
        ax_c.set_title(f"n = {n_plot}", fontsize=11)
        ax_c.legend(fontsize=7); ax_c.grid(True, alpha=0.3)
    plt.tight_layout(); save_fig(fig_cpu, "ex1-rse-cpu")

    # ── TGDBEK X-RSE vs Z-RSE for n=20 and n=80 ─────────────────────────────
    _zrse_data = {}
    for n_plot in [20, 80]:
        rng3 = np.random.default_rng(SEED)
        A_np  = rng3.standard_normal((M, n_plot, P)).astype(np.float32)
        Xs_np = rng3.standard_normal((n_plot, K, P)).astype(np.float32)
        A_t   = torch.from_numpy(A_np).to(device=device, dtype=dtype)
        Xs_t  = torch.from_numpy(Xs_np).to(device=device, dtype=dtype)
        B_clean = tp(A_t, Xs_t)
        g = torch.Generator(device=device); g.manual_seed(SEED)
        zeta = torch.randn(B_clean.shape, device=device, dtype=dtype, generator=g)
        eps  = NOISE * torch.linalg.norm(B_clean) / (torch.linalg.norm(zeta) + 1e-12) * zeta
        B_t   = B_clean + eps
        Xls_t = tpa(A_t, B_t, rcond=1e-12)
        (_, _, h_tg, h_z_tg, _), _ = tgdbek_with_z_rse(A_t, B_t, T, Xls_t,
                                                         eta=ETA, tol=1e-5, rcond=1e-12)
        _zrse_data[n_plot] = (h_tg, h_z_tg)

        # Save standalone single-panel figure for the paper
        fig_s, ax_s = plt.subplots(figsize=(6, 4))
        ax_s.semilogy(h_tg,   linewidth=2.0, color="tab:red", linestyle="-",
                      label=r"$\mathrm{RSE}^X$")
        ax_s.semilogy(h_z_tg, linewidth=2.0, color="tab:red", linestyle="--", alpha=0.7,
                      label=r"$\mathrm{RSE}^Z$")
        ax_s.axhline(1e-5, color="k", linestyle=":", linewidth=1, label="tol")
        ax_s.set_xlabel("Iteration", fontsize=12); ax_s.set_ylabel("RSE", fontsize=12)
        ax_s.set_title(f"TGDBEK  (n={n_plot})", fontsize=11)
        ax_s.legend(fontsize=9); ax_s.grid(True, alpha=0.3)
        plt.tight_layout(); save_fig(fig_s, f"ex1-z-rse-n{n_plot}")

    # Combined two-panel figure
    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
    for ax4, n_plot in zip(axes4, [20, 80]):
        h_tg, h_z_tg = _zrse_data[n_plot]
        ax4.semilogy(h_tg,   linewidth=2.0, color="tab:red", linestyle="-",
                     label=r"X-RSE  $\|\mathcal{X}^{(k)}-\mathcal{X}^*\|_F^2/\|\mathcal{X}^*\|_F^2$")
        ax4.semilogy(h_z_tg, linewidth=2.0, color="tab:red", linestyle="--", alpha=0.7,
                     label=r"Z-RSE  $\|\mathcal{Z}^{(k)}-\mathcal{Z}^*\|_F^2/\|\mathcal{Z}^*\|_F^2$")
        ax4.axhline(1e-5, color="k", linestyle=":", linewidth=1, label="tol=1e-5")
        ax4.set_xlabel("Iteration", fontsize=12); ax4.set_ylabel("RSE", fontsize=12)
        ax4.set_title(f"TGDBEK X-RSE and Z-RSE  (n={n_plot})", fontsize=11)
        ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)
    plt.tight_layout(); save_fig(fig4, "ex1-z-rse")

    # ── Empirical contraction rate β vs n ─────────────────────────────────────
    def _rate(hist):
        """Geometric mean per-iteration rate from log-linear fit on RSE history."""
        h = np.array(hist, dtype=float)
        h = h[h > 0]
        if len(h) < 2:
            return 1.0
        ks = np.arange(len(h))
        slope = np.polyfit(ks, np.log(h), 1)[0]
        return float(np.exp(slope))

    print("\n  Computing contraction rates vs n ...")
    rates = {m: [] for m in METHODS}
    rates["TGDBEK_Z"] = []
    for n_plot in N_vals:
        rng5 = np.random.default_rng(SEED)
        A_np  = rng5.standard_normal((M, n_plot, P)).astype(np.float32)
        Xs_np = rng5.standard_normal((n_plot, K, P)).astype(np.float32)
        A_t   = torch.from_numpy(A_np).to(device=device, dtype=dtype)
        Xs_t  = torch.from_numpy(Xs_np).to(device=device, dtype=dtype)
        B_clean = tp(A_t, Xs_t)
        g = torch.Generator(device=device); g.manual_seed(SEED)
        zeta = torch.randn(B_clean.shape, device=device, dtype=dtype, generator=g)
        eps  = NOISE * torch.linalg.norm(B_clean) / (torch.linalg.norm(zeta) + 1e-12) * zeta
        B_t   = B_clean + eps
        Xls_t = tpa(A_t, B_t, rcond=1e-12)
        row_p = make_partitions(M,      tau=TAU, sequential=True)
        col_p = make_partitions(n_plot, tau=TAU, sequential=True)

        for method, fn, kw in [
            ("TREK",   trek_algorithm,   dict(tol=1e-5, seed=SEED)),
            ("TREBK",  trebk_algorithm,  dict(row_partitions=row_p, col_partitions=col_p,
                                              tol=1e-5, seed=SEED)),
            ("TREGBK", tregbk_algorithm, dict(delta=DELTA, row_partitions=row_p,
                                              tol=1e-5, seed=SEED)),
        ]:
            (_, _, h, _), _ = fn(A_t, B_t, T, Xls_t, **kw)
            rates[method].append(_rate(h))

        (_, _, h_tg, h_z_tg, _), _ = tgdbek_with_z_rse(A_t, B_t, T, Xls_t,
                                                         eta=ETA, tol=1e-5, rcond=1e-12)
        rates["TGDBEK"].append(_rate(h_tg))
        rates["TGDBEK_Z"].append(_rate(h_z_tg))
        print(f"    n={n_plot}: " +
              "  ".join(f"{m}={rates[m][-1]:.4f}" for m in METHODS) +
              f"  TGDBEK_Z={rates['TGDBEK_Z'][-1]:.4f}")

    fig5, ax5 = plt.subplots(figsize=(7, 4))
    for m in METHODS:
        st = METHOD_STYLE[m]
        ax5.plot(N_vals, rates[m], label=m, linewidth=1.8, **st)
    st_tg = METHOD_STYLE["TGDBEK"]
    ax5.plot(N_vals, rates["TGDBEK_Z"],
             label=r"TGDBEK ($\hat\beta_Z$)",
             color=st_tg["color"], linestyle="--", linewidth=1.8,
             marker=st_tg["marker"], markersize=5, alpha=0.75)
    ax5.set_xlabel("$n$ (column dimension)", fontsize=12)
    ax5.set_ylabel(r"Empirical contraction rate $\hat\beta$", fontsize=12)
    ax5.set_title("Per-iteration contraction rate vs $n$", fontsize=11)
    ax5.legend(fontsize=9); ax5.grid(True, alpha=0.3)
    plt.tight_layout(); save_fig(fig5, "ex1-contraction-vs-n")

    print("  Example 4.1 complete.\n")
    print(f"  Figures saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
