#!/usr/bin/env python3
"""Example 4.3 — Color image deblurring (flower.jpg).

200×200×3, sigma=4, band=32, noise=1e-2, T=1000, eta=0.5
"""
import numpy as np
import matplotlib.pyplot as plt

from utils import (device, SEED, OUTDIR, IMGS_DIR, METHODS, METHOD_STYLE,
                   _HAVE_IMAGE, make_noise, make_blur_tensor, load_rgb_tensor,
                   run_image_experiment, save_image_grid, print_image_table, save_fig)
import os


def main():
    if not _HAVE_IMAGE:
        print("  Skipping Example 4.3 (scikit-image / imageio not available)")
        return

    print("\n" + "█"*65)
    print("  EXAMPLE 4.3  Color image deblurring (flower.jpg)")
    print("  200×200×3, σ=4, band=32, noise=1e-2, T=1000, η=0.5")
    print("█"*65)

    from tensor_toolbox.tensorLinalg import t_product as tp, t_pinv_apply as tpa

    IMG_PATH    = os.path.join(IMGS_DIR, "flower.jpg")
    N, P        = 200, 3
    SIGMA, BAND = 4.0, 32
    NOISE       = 1e-2
    T, ETA      = 1000, 0.5
    DELTA, TAU   = 0.7, 10
    RCOND_TGDBEK = 1e-2

    X_star  = load_rgb_tensor(IMG_PATH, N=N)
    A       = make_blur_tensor(N, P, sigma=SIGMA, band=BAND)
    B_clean = tp(A, X_star)
    B       = B_clean + make_noise(B_clean, NOISE, SEED)
    X_ls    = tpa(A, B, rcond=1e-3)

    print(f"  A shape: {tuple(A.shape)}, X*: {tuple(X_star.shape)}")

    results, orig_np, blur_np = run_image_experiment(
        A, B, X_star, X_ls, T, ETA, DELTA, TAU, RCOND_TGDBEK)

    print_image_table("TABLE 3  Color image deblurring (flower, 200×200×3)", results)
    save_image_grid(results, orig_np, blur_np, "ex3", is_color=True)

    # Extra paper figures (named differently from the grid saves)
    for method in METHODS:
        img = np.clip(results[method]["X"], 0, 1)
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(img); ax.axis("off")
        plt.tight_layout()
        save_fig(fig, f"ex3-{method.lower()}")

    fig2, ax2 = plt.subplots(figsize=(3, 3))
    ax2.imshow(orig_np); ax2.axis("off")
    plt.tight_layout(); save_fig(fig2, "ex3-origin-after-resh")

    fig3, ax3 = plt.subplots(figsize=(3, 3))
    ax3.imshow(np.clip(blur_np, 0, 1)); ax3.axis("off")
    plt.tight_layout(); save_fig(fig3, "ex3-blur")

    # Standalone convergence figure (X-RSE only)
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    for m in METHODS:
        h = results[m]["hist"]; st = METHOD_STYLE[m]
        ax4.semilogy(h, label=m, color=st["color"], linestyle=st["linestyle"],
                     linewidth=1.5, markevery=max(1, len(h)//20),
                     marker=st["marker"], markersize=4)
    ax4.axhline(1e-5, color="k", linestyle=":", linewidth=1, label="tol")
    ax4.set_xlabel("Iteration", fontsize=12); ax4.set_ylabel("RSE", fontsize=12)
    ax4.set_title("RSE vs iteration — color image deblurring", fontsize=11)
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)
    plt.tight_layout(); save_fig(fig4, "ex3-plot")

    # TGDBEK X-RSE and Z-RSE on same axes (paper fig ex3-plot-xz)
    h_x = results["TGDBEK"]["hist"]
    h_z = results["TGDBEK"]["z_hist"]
    fig5, ax5 = plt.subplots(figsize=(7, 5))
    ax5.semilogy(h_x, linewidth=2.0, color="tab:red", linestyle="-",
                 label=r"TGDBEK $\mathrm{RSE}^X$")
    ax5.semilogy(h_z, linewidth=2.0, color="tab:red", linestyle="--", alpha=0.7,
                 label=r"TGDBEK $\mathrm{RSE}^Z$")
    ax5.axhline(1e-5, color="k", linestyle=":", linewidth=1, label="tol")
    ax5.set_xlabel("Iteration", fontsize=12); ax5.set_ylabel("RSE", fontsize=12)
    ax5.set_title(r"$\mathrm{RSE}^X$ and $\mathrm{RSE}^Z$ — color image deblurring", fontsize=11)
    ax5.legend(fontsize=9); ax5.grid(True, alpha=0.3)
    plt.tight_layout(); save_fig(fig5, "ex3-plot-xz")

    print("  Example 4.3 complete.\n")
    print(f"  Figures saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
