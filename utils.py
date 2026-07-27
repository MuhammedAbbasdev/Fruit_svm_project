"""
utils.py
--------
Shared helpers used by train.py and evaluate.py:

  - Part A: build_splits() / extract_and_cache()  -> clean data/raw/<class>/*,
    split 70/15/15, cache Part-B feature vectors to data/processed/*.npz
  - StandardScaler                                 -> from-scratch feature
    standardization (fit on train only)
  - accuracy() / calibration_check()               -> evaluation metrics
  - plotting helpers                                -> training curves,
    confusion matrices, linear-vs-kernel comparison (saved under models/figures/)
  - load_trained_model() / classes / scaler loading -> shared by evaluate.py
"""

import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import extract_features_from_path

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
FIGURES_DIR = "models/figures"
SPLIT_SEED = 123
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15


# ---------------------------------------------------------------------------
# Part A: data cleaning + 70/15/15 split + feature caching
# ---------------------------------------------------------------------------
def build_splits():
    """
    Walks data/raw/<class_name>/*, drops any file OpenCV can't decode
    (logged as the "cleaning" step for the report), and splits the rest
    70% train / 15% val / 15% test per class.
    """
    classes = sorted([d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    rng = np.random.default_rng(SPLIT_SEED)

    splits = {"train": [], "val": [], "test": []}
    dropped = []

    for cls in classes:
        paths = sorted(glob.glob(os.path.join(RAW_DIR, cls, "*")))
        good_paths = []
        for p in paths:
            try:
                _ = extract_features_from_path(p)
                good_paths.append(p)
            except Exception as e:
                dropped.append((p, str(e)))
        good_paths = np.array(good_paths)
        rng.shuffle(good_paths)

        n = len(good_paths)
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))
        train_p, val_p, test_p = (good_paths[:n_train],
                                   good_paths[n_train:n_train + n_val],
                                   good_paths[n_train + n_val:])

        for p in train_p:
            splits["train"].append((p, class_to_idx[cls]))
        for p in val_p:
            splits["val"].append((p, class_to_idx[cls]))
        for p in test_p:
            splits["test"].append((p, class_to_idx[cls]))

        print(f"{cls:12s}: {n:4d} usable images -> train={len(train_p)}, val={len(val_p)}, test={len(test_p)}")

    if dropped:
        print(f"\nDropped {len(dropped)} unreadable/corrupt files:")
        for p, e in dropped:
            print(f"  {p}: {e}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(os.path.join(PROCESSED_DIR, "classes.json"), "w") as f:
        json.dump(classes, f)

    return splits, classes


def extract_and_cache(splits):
    """Extracts Part-B feature vectors for every image in every split and
    caches them to data/processed/{split}.npz so re-running train.py
    doesn't redo feature extraction from scratch every time."""
    from features import FEATURE_LENGTH
    for split_name, items in splits.items():
        n = len(items)
        X = np.zeros((n, FEATURE_LENGTH), dtype=np.float64)
        y = np.zeros(n, dtype=np.int64)
        for i, (path, label) in enumerate(items):
            X[i] = extract_features_from_path(path)
            y[i] = label
        np.savez(os.path.join(PROCESSED_DIR, f"{split_name}.npz"), X=X, y=y)
        print(f"cached {split_name}: X{X.shape}, y{y.shape}")


def load_split(name):
    d = np.load(f"{PROCESSED_DIR}/{name}.npz")
    return d["X"], d["y"]


def load_classes():
    with open(f"{PROCESSED_DIR}/classes.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# From-scratch feature standardization
# ---------------------------------------------------------------------------
class StandardScaler:
    """Zero-mean / unit-variance standardizer, fit on train only."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def save(self, path):
        np.savez(path, mean=self.mean_, std=self.std_)

    @staticmethod
    def load(path):
        d = np.load(path)
        s = StandardScaler()
        s.mean_, s.std_ = d["mean"], d["std"]
        return s


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def calibration_check(y_true, y_pred, confidences, title=""):
    correct = y_pred == y_true
    mean_conf_correct = confidences[correct].mean() if correct.any() else float("nan")
    mean_conf_wrong = confidences[~correct].mean() if (~correct).any() else float("nan")
    if title:
        print(f"  [{title}] mean confidence | correct preds = {mean_conf_correct:.3f}   "
              f"wrong preds = {mean_conf_wrong:.3f}")
    return mean_conf_correct, mean_conf_wrong


# ---------------------------------------------------------------------------
# Plotting (saved under models/figures/ for the report)
# ---------------------------------------------------------------------------
def plot_loss_curves(ovr_linear, classes, path):
    plt.figure(figsize=(7, 5))
    for k in ovr_linear.classes_:
        hist = ovr_linear.classifiers[k].loss_history
        epochs = [e for e, _ in hist]
        losses = [l for _, l in hist]
        plt.plot(epochs, losses, marker="o", markersize=3, label=classes[k])
    plt.xlabel("epoch")
    plt.ylabel("hinge loss objective  (1/2||w||^2 + C * mean hinge loss)")
    plt.title("Linear SVM training curves (One-vs-Rest, one line per class)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_confusion(y_true, y_pred, classes, path, title):
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(n), classes, rotation=45, ha="right")
    plt.yticks(range(n), classes)
    for i in range(n):
        for j in range(n):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_linear_vs_kernel(results, path):
    labels = ["train", "val", "test"]
    lin = [results["linear"][s] for s in labels]
    ker = [results["kernel"][s] for s in labels]
    x = np.arange(len(labels))
    w = 0.35
    plt.figure(figsize=(6, 4.5))
    plt.bar(x - w / 2, lin, w, label="Linear SVM (Part C)")
    plt.bar(x + w / 2, ker, w, label=f"Kernel SVM - {results['kernel_name']} (Part D)")
    plt.xticks(x, labels)
    plt.ylabel("accuracy")
    plt.ylim(0, 1.05)
    for i, v in enumerate(lin):
        plt.text(i - w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    for i, v in enumerate(ker):
        plt.text(i + w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    plt.title("Linear vs. Kernel SVM accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


# ---------------------------------------------------------------------------
# Model loading (shared by evaluate.py)
# ---------------------------------------------------------------------------
def load_trained_model(model_name=None):
    """
    model_name: 'linear', 'kernel', or None (use models/config.json's
    recorded best model). Returns (model, classes, scaler).
    """
    from svm import OneVsRestSVM  # local import avoids a circular import at module load time

    classes = load_classes_from_models()
    scaler = StandardScaler.load(f"{MODELS_DIR}/scaler.npz")

    if model_name is None:
        with open(f"{MODELS_DIR}/config.json") as f:
            model_name = json.load(f)["best_model"]

    fname = "linear_ovr_svm.pkl" if model_name == "linear" else "kernel_ovr_svm.pkl"
    model = OneVsRestSVM.load(os.path.join(MODELS_DIR, fname))
    return model, classes, scaler, model_name


def load_classes_from_models():
    with open(f"{MODELS_DIR}/classes.json") as f:
        return json.load(f)


def evaluate_directory_accuracy(model, classes, scaler, root_dir):
    """
    Evaluates accuracy on a directory laid out as root_dir/<class_name>/*.jpg,
    used to simulate the assignment's "instructor's hidden test set" protocol
    against a held-out folder that was never touched by preprocess.py's
    train/val/test split (e.g. a disjoint capture session / your own phone
    photos dropped in later). Returns (overall_acc, per_class_dict).
    """
    total, correct = 0, 0
    per_class = {}
    if not os.path.isdir(root_dir):
        return None, {}
    for cls in sorted(os.listdir(root_dir)):
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = [f for f in os.listdir(cls_dir)]
        c_correct = 0
        for f in files:
            try:
                feats = extract_features_from_path(os.path.join(cls_dir, f))
            except Exception:
                continue
            feats_s = scaler.transform(feats.reshape(1, -1))
            pred_classes, _, _ = model.predict_with_confidence(feats_s)
            pred_label = classes[pred_classes[0]]
            if pred_label == cls:
                c_correct += 1
                correct += 1
            total += 1
        per_class[cls] = {"correct": c_correct, "total": len(files)}
    overall_acc = correct / total if total else None
    return overall_acc, per_class
