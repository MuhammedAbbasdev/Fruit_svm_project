"""
kernels.py
----------
Hand-written kernel functions for the kernelized SVM (Module 07).

All kernels operate on batches of vectors (2D numpy arrays of shape
(n_samples, n_features)) and return a full Gram / kernel matrix of shape
(n_a, n_b), K[i, j] = k(x_i, z_j). Writing them in vectorized form (rather
than a python double for-loop) is what makes training on a few hundred
images per class tractable.

No sklearn.metrics.pairwise, no cvxopt kernels — every formula below is
implemented directly from the Module 07 definitions.
"""

import numpy as np


def linear_kernel(X, Z):
    """K(x, z) = x . z"""
    return X @ Z.T


def polynomial_kernel(X, Z, degree=3, coef0=1.0, gamma=1.0):
    """K(x, z) = (gamma * x.z + coef0) ** degree"""
    return (gamma * (X @ Z.T) + coef0) ** degree


def rbf_kernel(X, Z, gamma=None):
    """
    K(x, z) = exp(-gamma * ||x - z||^2)

    gamma here plays the role of 1 / (2 * sigma^2) from the assignment's
    K(x,z) = exp(-||x-z||^2 / (2*sigma^2)) formulation -- same kernel,
    just parameterized the way most SVM literature (and Module 07's
    bonus notes) writes it, so it's a 1-line conversion if you'd rather
    tune sigma directly: gamma = 1 / (2 * sigma**2).

    Uses the expansion ||x-z||^2 = ||x||^2 + ||z||^2 - 2 x.z so we get
    the full pairwise squared-distance matrix with matrix multiplies
    instead of a python loop over every pair.
    """
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    X_sq = np.sum(X ** 2, axis=1).reshape(-1, 1)      # (n_a, 1)
    Z_sq = np.sum(Z ** 2, axis=1).reshape(1, -1)      # (1, n_b)
    sq_dists = X_sq + Z_sq - 2.0 * (X @ Z.T)
    sq_dists = np.maximum(sq_dists, 0.0)  # guard against tiny negative fp noise
    return np.exp(-gamma * sq_dists)


KERNELS = {
    "linear": linear_kernel,
    "poly": polynomial_kernel,
    "rbf": rbf_kernel,
}


def get_kernel(name, **kwargs):
    """Return a kernel_fn(X, Z) with hyperparameters baked in via closure."""
    if name not in KERNELS:
        raise ValueError(f"Unknown kernel '{name}'. Choose from {list(KERNELS)}")
    base_fn = KERNELS[name]

    def kernel_fn(X, Z):
        return base_fn(X, Z, **kwargs)

    kernel_fn.name = name
    kernel_fn.params = kwargs
    return kernel_fn


if __name__ == "__main__":
    # tiny self-test
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 4))
    Z = rng.normal(size=(3, 4))
    print("linear:", linear_kernel(X, Z).shape)
    print("poly:", polynomial_kernel(X, Z, degree=2, coef0=1.0).shape)
    print("rbf:", rbf_kernel(X, Z, gamma=0.5).shape)
