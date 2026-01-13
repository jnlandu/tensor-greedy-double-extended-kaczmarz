"""
TGDBEK algorithm implementations.
"""

import time
import numpy as np
import torch

from tensor_toolbox.tensorLinalg import (
    t_product,
    t_transpose,
    t_pinv_apply,
)

from tgdbek.utils import rel_se


def tgdbek_algorithm_faithful(A, B, T, x_ls, eta=0.9, tol=1e-5, rcond=1e-12):
    """
    Faithful TGDBEK (per the latest screenshot with corrected J_k):

    Z-step:
      eps_z^k = eta * max_j  || (A_{:,j,:})^T * Z^k ||_F^2 / ||A_{:,j,:}||_F^2
      U_k = { j : ||(A_{:,j,:})^T * Z^k||_F^2 >= eps_z^k * ||A_{:,j,:}||_F^2 }
      Z^{k+1} = Z^k - A_{:,U_k,:} * (A_{:,U_k,:})^dagger * Z^k

    X-step (corrected):
      eps_x^k = eta * max_i  ||B_{i,:,:} - Z^{k+1}_{i,:,:} - A_{i,:,:}*X^k||_F^2 / ||A_{i,:,:}||_F^2
      J_k = { i : ||B_i - Z_i^{k+1} - A_i*X^k||_F^2 >= eps_x^k * ||A_i||_F^2 }
      X^{k+1} = X^k + (A_{J_k,:,:})^dagger * (B_{J_k,:,:} - Z^{k+1}_{J_k,:,:} - A_{J_k,:,:}*X^k)

    Parameters
    ----------
    A : (m, n, p) tensor
    B : (m, k, p) tensor
    T : int max iterations
    x_ls : (n, k, p) tensor reference (e.g. least-squares solution)
    eta : greedy parameter in (0,1]
    tol : stopping threshold on RSE
    rcond : rcond for torch.linalg.pinv

    Returns (same style you use)
    -------
    (X_np, iters, res_hist, x_hist_np), runtime
    """
    m, n, p = A.shape
    m_b, k, p_b = B.shape
    assert (m == m_b) and (p == p_b), "A:(m,n,p), B:(m,k,p) required"

    device = A.device
    dtype = A.dtype

    X = torch.zeros((n, k, p), dtype=dtype, device=device)
    Z = B.clone()

    # Precompute squared Frobenius norms of tensor-columns and tensor-rows
    # ||A_{:,j,:}||_F^2 over dims (0,2); shape (n,)
    col_norms_sq = torch.sum(A**2, dim=(0, 2)) + 1e-12
    # ||A_{i,:,:}||_F^2 over dims (1,2); shape (m,)
    row_norms_sq = torch.sum(A**2, dim=(1, 2)) + 1e-12

    res_hist = []
    x_hist = []

    res_hist.append(float(rel_se(X, x_ls)))
    x_hist.append(X.clone())

    t0 = time.time()
    with torch.no_grad():
        for iter_k in range(T):

            # =========================================================
            # Z-step: greedy set U_k (exactly as in the screenshot)
            # =========================================================

            trans_A = t_transpose(A)      # (n, m, p)
            trans_A_Z = t_product(trans_A, Z) # (n, m, p) * (m, k, p) = (n, k, p)
            scores_z = torch.sum(trans_A_Z**2, dim=(1, 2)) / col_norms_sq

            del trans_A   # to free memory
            del trans_A_Z  # to free memory

            eps_z = eta * torch.max(scores_z)
            U_k = torch.where(scores_z >= eps_z)[0]

            del eps_z
            del scores_z


            # Block projection: Z <- Z - A_U * (A_U)^dagger * Z
            A_U = A[:, U_k, :]                 # (m, |U|, p)
            W = t_pinv_apply(A_U, Z, rcond=rcond)  # (|U|, k, p) = (A_U)^dagger * Z
            Z = Z - t_product(A_U, W)          # (m, k, p)

            del W
            del A_U
            del U_k

            # =========================================================
            # X-step: corrected greedy set J_k (row-wise residual!)
            # =========================================================
            AX = t_product(A, X)               # (m, k, p)
            R = B - Z - AX                     # (m, k, p)

            del AX
            scores_x = torch.sum(R**2, dim=(1, 2)) / row_norms_sq
            eps_x = eta * torch.max(scores_x)
            J_k = torch.where(scores_x >= eps_x)[0]

            del R
            del scores_x
            del eps_x

            # Block correction: X <- X + (A_J)^dagger * (B_J - Z_J - A_J*X)
            A_J = A[J_k, :, :]  # (|J|, n, p)
            rhs = B[J_k, :, :] - Z[J_k, :, :] - t_product(A_J, X)  # (|J|, k, p)
            dX = t_pinv_apply(A_J, rhs, rcond=rcond)               # (n, k, p)
            X = X + dX

            del A_J
            del rhs
            del dX

            # =========================================================
            # Monitor with RSE to x_ls (your style)
            # =========================================================
            rse = rel_se(X, x_ls)
            res_hist.append(float(rse))
            x_hist.append(X.clone())
            
            if rse < tol:
                break

    runtime = time.time() - t0

    return (
        X.detach().cpu().numpy(),
        iter_k + 1,
        np.array(res_hist),
        np.array([x.detach().cpu().numpy() for x in x_hist])
    ), runtime