#!/usr/bin/env python3
"""Example 4.5 — Gray (MRI-like) image deblurring, Shepp-Logan 128×128×27.

128×128×27, sigma=4, band=64, noise=1e-3, T=1000, eta=0.5
"""
import numpy as np
import matplotlib.pyplot as plt

from utils import (device, SEED, OUTDIR, METHODS,
                   _HAVE_IMAGE, make_noise, make_blur_tensor, make_shepp_logan_tensor,
                   run_image_experiment, save_image_grid, print_image_table, save_fig)


def main():
    if not _HAVE_IMAGE:
        print("  Skipping Example 4.5 (scikit-image / imageio not available)")
        return

    print("\n" + "█"*65)
    print("  EXAMPLE 4.5  Gray MRI-like image deblurring (Shepp-Logan)")
    print("  128×128×27, σ=4, band=64, noise=1e-3, T=1000, η=0.5")
    print("█"*65)

    from tensor_toolbox.tensorLinalg import t_product as tp, t_pinv_apply as tpa

    N, P        = 128, 27
    SIGMA, BAND = 4.0, 64
    NOISE       = 1e-3
    T, ETA      = 1000, 0.5
    DELTA, TAU   = 0.7, 10
    RCOND_TGDBEK = 1e-2
    CENTRAL      = P // 2

    X_star  = make_shepp_logan_tensor(N=N, p=P)
    A       = make_blur_tensor(N, P, sigma=SIGMA, band=BAND)
    B_clean = tp(A, X_star)
    B       = B_clean + make_noise(B_clean, NOISE, SEED)
    X_ls    = tpa(A, B, rcond=1e-3)

    print(f"  A shape: {tuple(A.shape)}, X*: {tuple(X_star.shape)}")

    results, orig_np, blur_np = run_image_experiment(
        A, B, X_star, X_ls, T, ETA, DELTA, TAU, RCOND_TGDBEK)

    print_image_table("TABLE 5  Gray MRI image deblurring (128×128×27, noise=1e-3)", results)
    save_image_grid(results, orig_np, blur_np, "gray",
                    is_color=False, central_slice=CENTRAL)

    # Paper figures (central slice, grayscale)
    orig_slice = np.clip(orig_np[..., CENTRAL], 0, 1)
    blur_slice = np.clip(blur_np[..., CENTRAL], 0, 1)
    for arr, stem in [(orig_slice, "gray-original"), (blur_slice, "gray-blur")]:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(arr, cmap="gray"); ax.axis("off")
        plt.tight_layout(); save_fig(fig, stem)

    for method in METHODS:
        arr = np.clip(results[method]["X"][..., CENTRAL], 0, 1)
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(arr, cmap="gray"); ax.axis("off")
        plt.tight_layout(); save_fig(fig, f"gray-{method.lower()}")

    print("  Example 4.5 complete.\n")
    print(f"  Figures saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
