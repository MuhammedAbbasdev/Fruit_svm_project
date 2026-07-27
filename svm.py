"""
svm.py
------
From-scratch Support Vector Machine implementation for Module 06/07.

Everything here is built on top of raw numpy. No sklearn.svm.SVC,
no sklearn.multiclass, no cvxopt. Three pieces:

  1. LinearSVM   - primal soft-margin SVM, trained with (sub)gradient
                    descent on the hinge-loss objective:
                        min_w,b  1/2||w||^2 + C * sum_i max(0, 1 - y_i(w.x_i+b))
  2. KernelSVM   - kernelized SVM trained with the kernel-Pegasos
                    algorithm (Shalev-Shwartz et al.), which solves the
                    dual problem implicitly via stochastic sub-gradient
                    updates on dual coefficients alpha_i, using only
                    kernel evaluations K(x_i, x_j). This is what lets a
                    polynomial/RBF kernel bend the decision boundary
                    without ever materializing the high-dim feature map.
  3. OneVsRestSVM - trains K binary SVMs (linear or kernel), one per
                    fruit class, and combines them:
                        predict(x) = argmax_k  f_k(x)
                    Also fits a per-class Platt (sigmoid) calibrator so
                    raw decision scores become 0-100% confidences.
"""

import numpy as np
import pickle

from kernels import get_kernel


# ---------------------------------------------------------------------------
# 1. Linear SVM (primal, gradient descent on hinge loss)
# ---------------------------------------------------------------------------
class LinearSVM:
    """
    Soft-margin linear SVM, primal formulation:
        min_w,b  1/2 ||w||^2 + C * sum_i max(0, 1 - y_i (w.x_i + b))

    Trained with mini-batch sub-gradient descent -- for each sample in a
    batch, if y_i(w.x_i+b) < 1 it "violates the margin" and contributes
    -C*y_i*x_i to the gradient of w (and -C*y_i to the gradient of b);
    otherwise only the 1/2||w||^2 regularizer contributes to grad_w.
    """

    def __init__(self, C=1.0, lr=0.01, epochs=200, batch_size=32, seed=0):
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.w = None
        self.b = 0.0
        self.loss_history = []

    def _hinge_loss(self, X, y):
        margins = 1 - y * (X @ self.w + self.b)
        hinge = np.maximum(0, margins)
        return 0.5 * np.dot(self.w, self.w) + self.C * np.mean(hinge)

    def fit(self, X, y):
        """y must be +1 / -1 labels."""
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.loss_history = []

        for epoch in range(self.epochs):
            idx = rng.permutation(n)
            X_shuf, y_shuf = X[idx], y[idx]
            for start in range(0, n, self.batch_size):
                xb = X_shuf[start:start + self.batch_size]
                yb = y_shuf[start:start + self.batch_size]
                if len(xb) == 0:
                    continue
                margins = yb * (xb @ self.w + self.b)
                violate = margins < 1
                if np.any(violate):
                    grad_w = self.w - self.C * (yb[violate, None] * xb[violate]).sum(axis=0) / len(xb)
                    grad_b = -self.C * yb[violate].sum() / len(xb)
                else:
                    grad_w = self.w.copy()
                    grad_b = 0.0
                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b
            if epoch % max(1, self.epochs // 20) == 0 or epoch == self.epochs - 1:
                self.loss_history.append((epoch, self._hinge_loss(X, y)))
        return self

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))


# ---------------------------------------------------------------------------
# 2. Kernelized SVM (kernel Pegasos)
# ---------------------------------------------------------------------------
class KernelSVM:
    """
    Kernelized soft-margin SVM trained with the kernel-Pegasos algorithm.

    Instead of solving the primal over an explicit (possibly infinite
    dimensional) feature map, we maintain dual coefficients alpha_i and
    only ever touch the data through kernel evaluations K(x_i, x_j).
    This is the kernel trick in action: a polynomial or RBF kernel lets
    the decision boundary bend through a high-dimensional feature space
    that is never explicitly computed.

    Algorithm (per iteration t = 1..T):
        i_t   <- random training index
        score <- (1 / (lambda*t)) * sum_j alpha_j * y_j * K(x_j, x_it)
        if y_it * score < 1:      # margin violated -> update
            alpha_it += 1

    Final dual weights: c_j = alpha_j * y_j / (lambda * T)
        decision_function(x) = sum_j c_j * K(x_j, x)

    lambda (the L2 regularization strength) is tied to C via
    lambda = 1 / (C * n_samples), matching the primal C in LinearSVM so
    the two models are comparable in the report.
    """

    def __init__(self, kernel="rbf", C=1.0, iters=3000, kernel_params=None, seed=0):
        self.kernel_name = kernel
        self.C = C
        self.iters = iters
        self.kernel_params = kernel_params or {}
        self.seed = seed
        self.X_sv = None
        self.coef_ = None  # c_j = alpha_j * y_j / (lambda*T), only for kept support vectors

    @property
    def kernel_fn(self):
        # rebuilt on demand (not stored directly) so instances stay picklable --
        # a closure captured in get_kernel() cannot be pickled otherwise.
        return get_kernel(self.kernel_name, **self.kernel_params)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        lam = 1.0 / (self.C * n)
        K = self.kernel_fn(X, X)  # full Gram matrix, computed once
        alpha = np.zeros(n)

        for t in range(1, self.iters + 1):
            i = rng.integers(0, n)
            score = (alpha * y) @ K[:, i] / (lam * t)
            if y[i] * score < 1:
                alpha[i] += 1

        coef_full = alpha * y / (lam * self.iters)
        # keep only non-zero-influence points as "support vectors" (sparsify,
        # matches the assignment's requirement to save "support vectors + alpha")
        sv_mask = np.abs(coef_full) > 1e-12
        self.X_sv = X[sv_mask]
        self.coef_ = coef_full[sv_mask]
        self.n_support_ = sv_mask.sum()
        return self

    def decision_function(self, X):
        K_new = self.kernel_fn(self.X_sv, X)  # (n_sv, n_new)
        return self.coef_ @ K_new

    def predict(self, X):
        return np.sign(self.decision_function(X))


# ---------------------------------------------------------------------------
# 3. Platt scaling: turn a raw decision score into a calibrated probability
# ---------------------------------------------------------------------------
class PlattScaler:
    """
    Fits sigma(A*f(x) + B) to map a binary SVM's raw decision score f(x)
    into a probability, via simple gradient descent on the logistic
    negative log-likelihood (this is exactly the Module 03 logistic
    regression, just 1-D, on top of the frozen SVM scores).
    """

    def __init__(self, lr=0.05, epochs=500):
        self.lr = lr
        self.epochs = epochs
        self.A = -1.0
        self.B = 0.0

    def fit(self, scores, y01):
        """y01: labels in {0,1} (1 = this class)."""
        A, B = self.A, self.B
        n = len(scores)
        # target smoothing (Platt's recommended targets) to avoid infinite weights
        n_pos, n_neg = np.sum(y01 == 1), np.sum(y01 == 0)
        t = np.where(y01 == 1, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
        for _ in range(self.epochs):
            p = 1.0 / (1.0 + np.exp(-(A * scores + B)))
            p = np.clip(p, 1e-12, 1 - 1e-12)
            grad_A = np.mean((p - t) * scores)
            grad_B = np.mean(p - t)
            A -= self.lr * grad_A
            B -= self.lr * grad_B
        self.A, self.B = A, B
        return self

    def predict_proba(self, scores):
        return 1.0 / (1.0 + np.exp(-(self.A * scores + self.B)))


# ---------------------------------------------------------------------------
# 4. One-vs-Rest multi-class wrapper
# ---------------------------------------------------------------------------
class OneVsRestSVM:
    """
    Trains one binary SVM per class ("class k vs. everything else") and
    combines them at prediction time:
        predict(x) = argmax_k f_k(x)
    Confidence is produced by Platt-scaling each binary classifier's raw
    score into a probability using a held-out validation split, then
    normalizing the K calibrated probabilities into a distribution
    (renormalized softmax-style average) so they read as a 0-100% figure
    per prediction.
    """

    def __init__(self, mode="linear", class_names=None, **svm_kwargs):
        assert mode in ("linear", "kernel")
        self.mode = mode
        self.svm_kwargs = svm_kwargs
        self.class_names = class_names
        self.classifiers = {}
        self.calibrators = {}
        self.classes_ = None

    def _make_binary_svm(self):
        if self.mode == "linear":
            return LinearSVM(**self.svm_kwargs)
        else:
            return KernelSVM(**self.svm_kwargs)

    def fit(self, X, y, X_val=None, y_val=None, verbose=True):
        self.classes_ = np.unique(y)
        if X_val is None:
            X_val, y_val = X, y  # fallback, but a real val split is strongly preferred

        for k in self.classes_:
            y_bin = np.where(y == k, 1, -1)
            clf = self._make_binary_svm()
            clf.fit(X, y_bin)
            self.classifiers[k] = clf

            val_scores = clf.decision_function(X_val)
            y_bin_val01 = np.where(y_val == k, 1, 0)
            cal = PlattScaler().fit(val_scores, y_bin_val01)
            self.calibrators[k] = cal

            if verbose:
                name = self.class_names[k] if self.class_names else k
                train_acc = np.mean(np.sign(clf.decision_function(X)) == y_bin)
                print(f"  [OvR] class '{name}': binary train acc (this-vs-rest) = {train_acc:.3f}")
        return self

    def decision_function(self, X):
        """Raw (n_samples, n_classes) score matrix, columns in self.classes_ order."""
        scores = np.column_stack([self.classifiers[k].decision_function(X) for k in self.classes_])
        return scores

    def predict_proba(self, X):
        """
        Per-class Platt-calibrated probabilities, renormalized across
        classes so each row sums to 1 (i.e. reads as a genuine 0-100%
        confidence distribution over the K fruit classes).
        """
        raw_scores = self.decision_function(X)
        cal = np.column_stack([
            self.calibrators[k].predict_proba(raw_scores[:, i])
            for i, k in enumerate(self.classes_)
        ])
        row_sums = cal.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return cal / row_sums

    def predict(self, X):
        scores = self.decision_function(X)
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]

    def predict_with_confidence(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        pred_classes = self.classes_[idx]
        confidences = proba[np.arange(len(X)), idx]
        return pred_classes, confidences, proba

    # -- persistence -----------------------------------------------------
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
