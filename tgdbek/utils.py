"""
Utility functions for tensor computations.
"""

from tensor_toolbox.tensorLinalg import  t_frobenius_norm


def rel_se(X, X_ref):
    """
    Compute relative solution error.

    Paramaters:
    -----------
    X:  torch.tensor. 3d tensor of shape (n, k, p) 
    X_ref: (n, k, p): reference tensor or leas-squre solution

    Returns:
    --------
    rse: float. relative solution error.

    Example:
    --------
    >>> X = torch.randn(80, 4, 8)
    >>> X_ref = torch.randn(80, 4, 8)
    >>> rse = rel_se(X, X_ref)
    >>> print(f"Relative solution error: {rse:.6f}")
    Relative solution error: 1.234567

    """

    diff = X - X_ref
    frob_diff = t_frobenius_norm(diff)
    frob_ref = t_frobenius_norm(X_ref)
    rse = frob_diff / (frob_ref + 1e-12)
    return rse
