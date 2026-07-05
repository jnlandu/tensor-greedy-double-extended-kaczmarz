"""
utils.py — Shared setup, helpers, and plotting for TGDBEK experiments.

Imports this module to get: device, dtype, SEED, OUTDIR, SS_DATA, IMGS_DIR,
METHODS, METHOD_STYLE, _HAVE_IMAGE, and all helper functions.
"""

import sys, os, time, warnings
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ssgetpy
from scipy.io import mmread

# ── Paths ──────────────────────────────────────────────────────────────────────
EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_TGDBEK     = os.path.dirname(EXPERIMENTS_DIR)
SS_DATA         = os.path.join(EXPERIMENTS_DIR, "data", "suitesparse")
IMGS_DIR        = os.path.join(REPO_TGDBEK, "references", "imgs")
OUTDIR          = os.path.join(EXPERIMENTS_DIR, "figures")
SEED            = 1234

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(SS_DATA, exist_ok=True)
sys.path.insert(0, REPO_TGDBEK)

# ── Device ─────────────────────────────────────────────────────────────────────
device = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available()         else
          torch.device("cpu"))
dtype = torch.float32

# ── Optional image libs (Ex 4.3–4.5) ──────────────────────────────────────────
try:
    import imageio.v3 as iio
    from skimage import transform as sk_transform, img_as_float32, data as sk_data
    from skimage.transform import rotate as sk_rotate
    from skimage.metrics import peak_signal_noise_ratio as psnr, \
                                structural_similarity as ssim
    _HAVE_IMAGE = True
except ImportError:
    iio = sk_transform = img_as_float32 = sk_data = sk_rotate = None
    psnr = ssim = None
    _HAVE_IMAGE = False
    warnings.warn("scikit-image / imageio not found; image examples will be skipped.")

# ── Algorithm imports ──────────────────────────────────────────────────────────
from tgdbek.methods import tgdbek_algorithm_faithful
from trk_algorithms.methods import trek_algorithm, trebk_algorithm, tregbk_algorithm
from trk_algorithms.utils import make_partitions, rel_se

# ── Method styles ──────────────────────────────────────────────────────────────
METHODS = ["TREK", "TREBK", "TREGBK", "TGDBEK"]
METHOD_STYLE = {
    "TREK":   dict(color="tab:blue",   linestyle="-",  marker="o"),
    "TREBK":  dict(color="tab:orange", linestyle="--", marker="s"),
    "TREGBK": dict(color="tab:green",  linestyle="-.", marker="^"),
    "TGDBEK": dict(color="tab:red",    linestyle="-",  marker="*"),
}


# ══════════════════════════════════════════════════════════════════════════════
# Tensor problem builders
# ══════════════════════════════════════════════════════════════════════════════

def make_tensor_problem(m, n, p, K, noise, rng):
    """Random Gaussian overdetermined tensor system (B = A*X* + eps)."""
    from tensor_toolbox.tensorLinalg import t_product as tp, t_pinv_apply as tpa
    A_np  = rng.standard_normal((m, n, p)).astype(np.float32)
    Xs_np = rng.standard_normal((n, K, p)).astype(np.float32)
    A  = torch.from_numpy(A_np).to(device=device, dtype=dtype)
    Xs = torch.from_numpy(Xs_np).to(device=device, dtype=dtype)
    B_clean = tp(A, Xs)
    g = torch.Generator(device=device); g.manual_seed(SEED)
    zeta = torch.randn(B_clean.shape, device=device, dtype=dtype, generator=g)
    eps  = noise * torch.linalg.norm(B_clean) / (torch.linalg.norm(zeta) + 1e-12) * zeta
    B    = B_clean + eps
    X_ls = tpa(A, B, rcond=1e-12)
    return A, B, X_ls


def sparse_mtx_to_tensor(name, N2, N3, fill_cols=None):
    """Download (if needed) and load a SuiteSparse matrix as an (m, N2, N3) tensor."""
    ssgetpy.fetch(name, location=SS_DATA)
    path  = os.path.join(SS_DATA, name, f"{name}.mtx")
    M_sp  = mmread(path)
    M     = np.array(M_sp.todense(), dtype=np.float32)
    m     = M.shape[0]
    ncols = N2 * N3 if fill_cols is None else fill_cols
    M_sub = M[:, :ncols]
    if M_sub.shape[1] < N2 * N3:
        pad   = np.zeros((m, N2 * N3 - M_sub.shape[1]), dtype=np.float32)
        M_sub = np.concatenate([M_sub, pad], axis=1)
    A = torch.from_numpy(M_sub.reshape((m, N2, N3), order="F")).to(device=device, dtype=dtype)
    return A


def make_noise(B_clean, noise_level, seed):
    g = torch.Generator(device=B_clean.device); g.manual_seed(seed)
    zeta = torch.randn(B_clean.shape, device=B_clean.device, dtype=B_clean.dtype, generator=g)
    zeta = zeta / (torch.linalg.norm(zeta) + 1e-12)
    return noise_level * torch.linalg.norm(B_clean) * zeta


def tgdbek_with_z_rse(A, B, T, x_ls, eta=0.9, tol=1e-5, rcond=1e-12):
    """Run TGDBEK tracking both X-RSE and Z-RSE (Z* = B - A·x_ls)."""
    from tensor_toolbox.tensorLinalg import (
        t_product as _tp, t_pinv_apply as _tpa,
        t_transpose as _tt, t_frobenius_norm as _tfn,
    )
    Z_star  = B - _tp(A, x_ls)
    z_denom = float(_tfn(Z_star) ** 2) + 1e-12

    _, n, p = A.shape
    K = x_ls.shape[1]
    X = torch.zeros((n, K, p), device=A.device, dtype=A.dtype)
    Z = B.clone()
    col_sq = torch.sum(A**2, dim=(0, 2)) + 1e-12
    row_sq = torch.sum(A**2, dim=(1, 2)) + 1e-12

    x_hist = [float(rel_se(X, x_ls))]
    z_hist = [float(_tfn(Z - Z_star) ** 2) / z_denom]
    time_hist = [0.0]

    t0 = time.time()
    with torch.no_grad():
        for k in range(T):
            tA  = _tt(A); tAZ = _tp(tA, Z)
            s_z = torch.sum(tAZ**2, dim=(1, 2)) / col_sq
            del tA, tAZ
            U = torch.where(s_z >= eta * torch.max(s_z))[0]; del s_z
            Au = A[:, U, :]
            Z  = Z - _tp(Au, _tpa(Au, Z, rcond=rcond))
            del Au, U

            R   = B - Z - _tp(A, X)
            s_x = torch.sum(R**2, dim=(1, 2)) / row_sq
            J   = torch.where(s_x >= eta * torch.max(s_x))[0]; del R, s_x
            Aj  = A[J, :, :]
            rhs = B[J,:,:] - Z[J,:,:] - _tp(Aj, X)
            X   = X + _tpa(Aj, rhs, rcond=rcond); del Aj, rhs, J

            xr = float(rel_se(X, x_ls))
            zr = float(_tfn(Z - Z_star) ** 2) / z_denom
            x_hist.append(xr); z_hist.append(zr)
            time_hist.append(time.time() - t0)
            if xr < tol:
                break

    return (X.cpu().numpy(), k + 1,
            np.array(x_hist), np.array(z_hist), np.array(time_hist)), time.time() - t0


# ══════════════════════════════════════════════════════════════════════════════
# Method runners
# ══════════════════════════════════════════════════════════════════════════════

def run_methods(A, B, X_ls, T, eta, delta, tau,
                rcond_tgdbek=1e-12, n_trials=1, seed=SEED):
    """Run TREK, TREBK, TREGBK, TGDBEK; optionally average over n_trials."""
    m, n, _ = A.shape
    row_parts = make_partitions(m, tau=tau, sequential=True)
    col_parts = make_partitions(n, tau=tau, sequential=True)

    def _once(s):
        (_, k, h, _), t = trek_algorithm(A, B, T, X_ls, tol=1e-5, seed=s)
        r = {"TREK": dict(it=k, cpu=t, rse=float(h[-1]), hist=h)}

        (_, k, h, _), t = trebk_algorithm(A, B, T, X_ls,
            row_partitions=row_parts, col_partitions=col_parts, tol=1e-5, seed=s)
        r["TREBK"] = dict(it=k, cpu=t, rse=float(h[-1]), hist=h)

        (_, k, h, _), t = tregbk_algorithm(A, B, T, X_ls, delta=delta,
            row_partitions=row_parts, tol=1e-5, rcond=1e-12, seed=s)
        r["TREGBK"] = dict(it=k, cpu=t, rse=float(h[-1]), hist=h)

        (Xtg, k, h, _), t = tgdbek_algorithm_faithful(A, B, T, X_ls,
            eta=eta, tol=1e-5, rcond=rcond_tgdbek)
        r["TGDBEK"] = dict(it=k, cpu=t, rse=float(h[-1]), hist=h, X=Xtg)
        return r

    if n_trials == 1:
        return _once(seed)

    all_res = [_once(seed + i) for i in range(n_trials)]
    avg = {}
    for m_ in all_res[0]:
        avg[m_] = dict(
            it  = np.mean([r[m_]["it"]  for r in all_res]),
            cpu = np.mean([r[m_]["cpu"] for r in all_res]),
            rse = np.mean([r[m_]["rse"] for r in all_res]),
            hist= all_res[0][m_]["hist"],
        )
    return avg


def run_image_experiment(A, B, X_star, X_ls, T, eta, delta, tau, rcond_tgdbek):
    """Run all four methods on an image deblurring system; return results dict."""
    from tensor_toolbox.tensorLinalg import t_product as tp, t_pinv_apply as tpa

    m, n, p = A.shape
    row_parts = make_partitions(m, tau=tau, sequential=True)
    col_parts = make_partitions(n, tau=tau, sequential=True)

    print("  Running TREK ...")
    (X_tr,  k_tr,  h_tr,  _),     t_tr  = trek_algorithm(A, B, T, X_ls, tol=1e-5, seed=SEED)
    print("  Running TREBK ...")
    (X_tb,  k_tb,  h_tb,  _),     t_tb  = trebk_algorithm(A, B, T, X_ls,
        row_partitions=row_parts, col_partitions=col_parts, tol=1e-5, seed=SEED)
    print("  Running TREGBK ...")
    (X_tg2, k_tg2, h_tg2, _),     t_tg2 = tregbk_algorithm(A, B, T, X_ls, delta=delta,
        row_partitions=row_parts, tol=1e-5, rcond=1e-12, seed=SEED)
    print("  Running TGDBEK ...")
    (X_tg,  k_tg,  h_tg,  h_z_tg, _), t_tg = tgdbek_with_z_rse(
        A, B, T, X_ls, eta=eta, tol=1e-5, rcond=rcond_tgdbek)

    results = {
        "TREK":   dict(X=X_tr,  it=k_tr,  cpu=t_tr,  hist=h_tr),
        "TREBK":  dict(X=X_tb,  it=k_tb,  cpu=t_tb,  hist=h_tb),
        "TREGBK": dict(X=X_tg2, it=k_tg2, cpu=t_tg2, hist=h_tg2),
        "TGDBEK": dict(X=X_tg,  it=k_tg,  cpu=t_tg,  hist=h_tg, z_hist=h_z_tg),
    }

    orig_np = np.clip(X_star.detach().cpu().numpy(), 0.0, 1.0)
    blur_np = np.clip(B.detach().cpu().numpy(), 0.0, 1.0)
    is_color = (p <= 4)

    for method, r in results.items():
        Xrec = np.clip(r["X"], 0.0, 1.0)
        if is_color:
            r["psnr"] = float(psnr(orig_np, Xrec, data_range=1.0))
            r["ssim"] = float(ssim(orig_np, Xrec, channel_axis=-1, data_range=1.0))
        else:
            sl = p // 2
            r["psnr"] = float(psnr(orig_np[..., sl], Xrec[..., sl], data_range=1.0))
            r["ssim"] = float(ssim(orig_np[..., sl], Xrec[..., sl], data_range=1.0))
        r["rse"] = float(r["hist"][-1])

    return results, orig_np, blur_np


# ══════════════════════════════════════════════════════════════════════════════
# Plotting & printing helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_fig(fig, stem, subdir=""):
    dest = os.path.join(OUTDIR, subdir)
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, f"{stem}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def print_table(title, rows):
    """rows: list of (Method, IT, CPU, RSE)"""
    print(f"\n{'='*65}\n  {title}\n{'='*65}")
    print(f"  {'Method':<10} {'IT':>8} {'CPU (s)':>10} {'RSE':>14}")
    print(f"  {'-'*46}")
    for method, it, cpu, rse in rows:
        it_str  = f"{it:.0f}" if isinstance(it, float) else str(it)
        rse_str = f"{rse:.3e}" if rse == rse else "NaN"
        print(f"  {method:<10} {it_str:>8} {cpu:>10.3f} {rse_str:>14}")
    print(f"{'='*65}\n")


def print_image_table(title, results):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")
    print(f"  {'Method':<10} {'IT':>6} {'CPU (s)':>10} {'RSE':>12} "
          f"{'PSNR (dB)':>12} {'SSIM':>8}")
    print(f"  {'-'*60}")
    for method in METHODS:
        r = results[method]
        print(f"  {method:<10} {r['it']:>6} {r['cpu']:>10.3f} "
              f"{r['rse']:>12.3e} {r['psnr']:>12.2f} {r['ssim']:>8.4f}")
    print(f"{'='*70}\n")


def save_image_grid(results, orig_np, blur_np, name, is_color, central_slice=None):
    """Save summary grid, individual images, and convergence plot."""
    def _show(ax, img, title):
        disp = np.clip(img if is_color else img[..., central_slice], 0, 1)
        ax.imshow(disp, cmap=None if is_color else "gray")
        ax.set_title(title, fontsize=8); ax.axis("off")

    # Summary grid
    fig, axes = plt.subplots(1, 2 + len(METHODS), figsize=(3 * (2 + len(METHODS)), 3))
    _show(axes[0], orig_np, "Original (X*)")
    _show(axes[1], blur_np, "Blurred + noise")
    for ax, m in zip(axes[2:], METHODS):
        r = results[m]
        _show(ax, np.clip(r["X"], 0, 1), f"{m}\nIT={r['it']} RSE={r['rse']:.1e}")
    plt.tight_layout(); save_fig(fig, f"{name}-summary")

    # Individual images
    for arr, stem in [(orig_np, f"{name}-original"), (blur_np, f"{name}-blur")]:
        fig2, ax2 = plt.subplots(figsize=(3, 3))
        _show(ax2, arr, ""); plt.tight_layout(); save_fig(fig2, stem)

    for m in METHODS:
        fig3, ax3 = plt.subplots(figsize=(3, 3))
        _show(ax3, np.clip(results[m]["X"], 0, 1), m)
        plt.tight_layout(); save_fig(fig3, f"{name}-{m.lower()}")

    # Convergence: X-RSE (all methods) + Z-RSE (TGDBEK)
    fig4, (ax4, ax4z) = plt.subplots(1, 2, figsize=(13, 5))
    for m in METHODS:
        h = results[m]["hist"]; st = METHOD_STYLE[m]
        ax4.semilogy(h, label=m, color=st["color"], linestyle=st["linestyle"],
                     markevery=max(1, len(h)//20), marker=st["marker"],
                     markersize=5, linewidth=1.5)
    ax4.axhline(1e-5, color="k", linestyle=":", linewidth=1, label="tol=1e-5")
    ax4.set_xlabel("Iteration", fontsize=12)
    ax4.set_ylabel(r"RSE  ($\|\mathcal{X}^{(k)}-\mathcal{X}^*\|_F^2/\|\mathcal{X}^*\|_F^2$)", fontsize=10)
    ax4.set_title(f"X-iterate RSE — {name.upper()}", fontsize=11)
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)

    hz = results["TGDBEK"].get("z_hist")
    if hz is not None:
        ax4z.semilogy(hz, label="TGDBEK Z-RSE",
                      color=METHOD_STYLE["TGDBEK"]["color"], linestyle="--", linewidth=2.0)
    ax4z.axhline(1e-5, color="k", linestyle=":", linewidth=1, label="tol=1e-5")
    ax4z.set_xlabel("Iteration", fontsize=12)
    ax4z.set_ylabel(r"Z-RSE  ($\|\mathcal{Z}^{(k)}-\mathcal{Z}^*\|_F^2/\|\mathcal{Z}^*\|_F^2$)", fontsize=10)
    ax4z.set_title(f"Z-iterate RSE (TGDBEK) — {name.upper()}", fontsize=11)
    ax4z.legend(fontsize=9); ax4z.grid(True, alpha=0.3)
    plt.tight_layout(); save_fig(fig4, f"{name}-convergence")


# ══════════════════════════════════════════════════════════════════════════════
# Image tensor builders (used by Ex 4.3–4.5)
# ══════════════════════════════════════════════════════════════════════════════

def _toeplitz_torch(c, r):
    c, r = c.flatten(), r.flatten()
    m, n = c.numel(), r.numel()
    i = torch.arange(m, device=c.device)[:, None]
    j = torch.arange(n, device=c.device)[None, :]
    diff = i - j
    return torch.where(diff >= 0, c[diff], r[-diff])


def make_blur_tensor(N, p, sigma=4.0, band=32):
    t = torch.arange(N, device=device, dtype=dtype)
    z = torch.zeros(N, device=device, dtype=dtype)
    z[:band] = torch.exp(-(t[:band] ** 2) / (2 * sigma ** 2))
    c  = torch.cat([z[:1], torch.flip(z[1:], dims=[0])], dim=0)
    A0 = _toeplitz_torch(c, z)
    A0 = (1 / (sigma * torch.pi)) * A0 / (A0.sum(dim=1, keepdim=True) + 1e-12)
    A  = torch.zeros((N, N, p), device=device, dtype=dtype)
    A[:, :, 0] = A0
    return A


def load_rgb_tensor(path, N=200):
    img = iio.imread(path)
    img = img_as_float32(img)
    img = sk_transform.resize(img, (N, N), anti_aliasing=True,
                              preserve_range=True).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    return torch.from_numpy(img).to(device=device, dtype=dtype)


def make_shepp_logan_tensor(N=128, p=27):
    base   = sk_data.shepp_logan_phantom().astype(np.float32)
    base   = sk_transform.resize(base, (N, N), anti_aliasing=True).astype(np.float32)
    base   = np.clip(base, 0.0, 1.0)
    angles = np.linspace(-8, 8, p)
    scales = 0.95 + 0.1 * np.sin(np.linspace(0, 2 * np.pi, p))
    vol    = np.zeros((N, N, p), dtype=np.float32)
    for k in range(p):
        sl = sk_rotate(base, float(angles[k]), resize=False,
                       mode="edge", preserve_range=True).astype(np.float32)
        vol[:, :, k] = np.clip(scales[k] * sl, 0.0, 1.0)
    return torch.from_numpy(vol).to(device=device, dtype=dtype)
