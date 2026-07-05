#!/usr/bin/env python3
"""Example 4.4 — High-iteration image deblurring (astronaut, scikit-image built-in).

200×200×3, sigma=4, band=32, noise=1e-1, T=3000, eta=0.5
"""
import numpy as np
import torch

from utils import (device, dtype, SEED, OUTDIR,
                   _HAVE_IMAGE, sk_data, sk_transform,
                   make_noise, make_blur_tensor,
                   run_image_experiment, save_image_grid, print_image_table)


def main():
    if not _HAVE_IMAGE:
        print("  Skipping Example 4.4 (scikit-image / imageio not available)")
        return

    print("\n" + "█"*65)
    print("  EXAMPLE 4.4  High-iteration image deblurring (astronaut)")
    print("  200×200×3, σ=4, band=32, noise=1e-1, T=3000, η=0.5")
    print("█"*65)

    from tensor_toolbox.tensorLinalg import t_product as tp, t_pinv_apply as tpa

    N, P        = 200, 3
    SIGMA, BAND = 4.0, 32
    NOISE       = 1e-1
    T, ETA      = 3000, 0.5
    DELTA, TAU  = 0.7, 10
    RCOND_TGDBEK = 1e-2

    # Load astronaut (512×512×3 uint8), resize to N×N
    img = sk_data.astronaut().astype(np.float32) / 255.0
    img = sk_transform.resize(img, (N, N), anti_aliasing=True,
                              preserve_range=True).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    X_star = torch.from_numpy(img).to(device=device, dtype=dtype)

    A       = make_blur_tensor(N, P, sigma=SIGMA, band=BAND)
    B_clean = tp(A, X_star)
    B       = B_clean + make_noise(B_clean, NOISE, SEED)
    X_ls    = tpa(A, B, rcond=1e-3)

    print(f"  A shape: {tuple(A.shape)}, X*: {tuple(X_star.shape)}")

    results, orig_np, blur_np = run_image_experiment(
        A, B, X_star, X_ls, T, ETA, DELTA, TAU, RCOND_TGDBEK)

    print_image_table("TABLE 4  High-iteration image deblurring (astronaut, 200×200×3, noise=1e-1)", results)
    save_image_grid(results, orig_np, blur_np, "ex4", is_color=True)

    print("  Example 4.4 complete.\n")
    print(f"  Figures saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
