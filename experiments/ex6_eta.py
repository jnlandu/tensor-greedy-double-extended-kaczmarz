#!/usr/bin/env python3
"""Example 4.6 — Effect of greedy threshold eta on TGDBEK.

A ∈ R^{500×20×10}, noise=1e-2, T=2000, tol=1e-6, 20 trials
eta ∈ {0.1, 0.2, ..., 1.0}
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import (device, dtype, SEED, OUTDIR,
                   tgdbek_algorithm_faithful, save_fig)


def main():
    print("\n" + "█"*65)
    print("  EXAMPLE 4.6  Effect of η on TGDBEK")
    print("  A ∈ R^{500×20×10}, noise=1e-2, T=2000, tol=1e-6, 20 trials")
    print("█"*65)

    from tensor_toolbox.tensorLinalg import t_product as tp, t_pinv_apply as tpa

    M, N, P, K = 500, 20, 10, 10
    NOISE, T   = 1e-2, 2000
    N_TRIALS   = 20
    ETA_VALS   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    rng = np.random.default_rng(SEED)
    eta_it_mean  = []
    eta_cpu_mean = []

    for eta in ETA_VALS:
        it_list, cpu_list = [], []
        for trial in range(N_TRIALS):
            A_np  = rng.standard_normal((M, N, P)).astype(np.float32)
            Xs_np = rng.standard_normal((N, K, P)).astype(np.float32)
            A_t   = torch.from_numpy(A_np).to(device=device, dtype=dtype)
            Xs_t  = torch.from_numpy(Xs_np).to(device=device, dtype=dtype)
            B_clean = tp(A_t, Xs_t)
            g = torch.Generator(device=device); g.manual_seed(SEED + trial)
            zeta = torch.randn(B_clean.shape, device=device, dtype=dtype, generator=g)
            eps  = NOISE * torch.linalg.norm(B_clean) / (torch.linalg.norm(zeta) + 1e-12) * zeta
            B_t  = B_clean + eps
            Xls_t = tpa(A_t, B_t, rcond=1e-12)

            (_, k, _, _), t = tgdbek_algorithm_faithful(A_t, B_t, T, Xls_t,
                                                         eta=eta, tol=1e-6, rcond=1e-12)
            it_list.append(k); cpu_list.append(t)

        eta_it_mean.append(np.mean(it_list))
        eta_cpu_mean.append(np.mean(cpu_list))
        print(f"  η={eta:.1f}  IT={eta_it_mean[-1]:.0f}  CPU={eta_cpu_mean[-1]:.3f}s")

    print(f"\n{'='*45}")
    print("  TABLE 6  Effect of greedy threshold η")
    print(f"  {'η':>5}  {'IT':>8}  {'CPU (s)':>10}")
    print(f"  {'-'*28}")
    for eta, it, cpu in zip(ETA_VALS, eta_it_mean, eta_cpu_mean):
        print(f"  {eta:>5.1f}  {it:>8.0f}  {cpu:>10.3f}")
    print(f"{'='*45}\n")

    # Combined figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, y, ylabel, title in zip(axes,
                                     [eta_it_mean, eta_cpu_mean],
                                     ["Iterations (IT)", "CPU time (s)"],
                                     ["IT vs η", "CPU vs η"]):
        ax.plot(ETA_VALS, y, "o-", color="tab:red", linewidth=1.8, markersize=6)
        ax.set_xlabel("η", fontsize=12); ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout(); save_fig(fig, "eta-convergence")

    # Separate paper figures
    for y, ylabel, stem in [(eta_it_mean,  "Iterations (IT)", "eta-IT"),
                             (eta_cpu_mean, "CPU time (s)",    "eta-cpu")]:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(ETA_VALS, y, "o-", color="tab:red", linewidth=1.8, markersize=6)
        ax2.set_xlabel("η", fontsize=12); ax2.set_ylabel(ylabel, fontsize=12)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout(); save_fig(fig2, stem)

    print("  Example 4.6 complete.\n")
    print(f"  Figures saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
