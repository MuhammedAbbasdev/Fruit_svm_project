"""
train.py
--------
Full pipeline, Parts A -> E:
  1. Part A: clean data/raw/<class>/*, split 70/15/15, cache feature vectors
     (utils.build_splits / utils.extract_and_cache).
  2. Standardize features (fit on train only).
  3. Part C: train a from-scratch linear One-vs-Rest SVM.
  4. Part D: train a from-scratch kernelized (RBF) One-vs-Rest SVM via
     kernel-Pegasos.
  5. Part E: calibrate confidences with Platt scaling on the validation split.
  6. Report train/val/test accuracy + a calibration check, save the models
     evaluate.py needs, and save figures for the report under models/figures/.
"""

import json
import os
import time
import numpy as np

from svm import OneVsRestSVM
import utils

MODELS_DIR = utils.MODELS_DIR
FIGURES_DIR = utils.FIGURES_DIR


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ---------------- Part A ----------------
    print("=== Part A: cleaning + splitting data/raw/ ===")
    splits, classes = utils.build_splits()
    utils.extract_and_cache(splits)

    X_train, y_train = utils.load_split("train")
    X_val, y_val = utils.load_split("val")
    X_test, y_test = utils.load_split("test")
    print(f"\ntrain={X_train.shape}, val={X_val.shape}, test={X_test.shape}, classes={classes}")

    scaler = utils.StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    results = {}

    # ---------------- Part C: Linear SVM (OvR) ----------------
    print("\n=== Part C: Linear SVM (One-vs-Rest, from-scratch gradient descent) ===")
    t0 = time.time()
    ovr_linear = OneVsRestSVM(mode="linear", class_names=classes,
                               C=1.0, lr=0.02, epochs=150, batch_size=32, seed=0)
    ovr_linear.fit(X_train_s, y_train, X_val=X_val_s, y_val=y_val)
    print(f"linear OvR trained in {time.time()-t0:.1f}s")

    pred_tr, conf_tr, _ = ovr_linear.predict_with_confidence(X_train_s)
    pred_val, conf_val, _ = ovr_linear.predict_with_confidence(X_val_s)
    pred_test, conf_test, _ = ovr_linear.predict_with_confidence(X_test_s)
    acc_tr, acc_val, acc_test = (utils.accuracy(y_train, pred_tr),
                                  utils.accuracy(y_val, pred_val),
                                  utils.accuracy(y_test, pred_test))
    print(f"Linear SVM accuracy -> train={acc_tr:.3f}  val={acc_val:.3f}  test={acc_test:.3f}")
    utils.calibration_check(y_test, pred_test, conf_test, "Linear/test")
    results["linear"] = {"train": acc_tr, "val": acc_val, "test": acc_test}

    utils.plot_loss_curves(ovr_linear, classes, f"{FIGURES_DIR}/linear_training_curves.png")
    utils.plot_confusion(y_test, pred_test, classes, f"{FIGURES_DIR}/linear_confusion_test.png",
                          "Linear SVM - Test confusion matrix")

    # ---------------- Part D: Kernel SVM (OvR) ----------------
    kernel_name = "rbf"
    print(f"\n=== Part D: Kernel SVM (OvR, kernel-Pegasos, kernel={kernel_name}) ===")
    t0 = time.time()
    ovr_kernel = OneVsRestSVM(mode="kernel", class_names=classes,
                               kernel=kernel_name, C=5.0, iters=4000,
                               kernel_params={"gamma": 1.0 / X_train_s.shape[1]}, seed=0)
    ovr_kernel.fit(X_train_s, y_train, X_val=X_val_s, y_val=y_val)
    print(f"kernel OvR trained in {time.time()-t0:.1f}s")

    pred_tr_k, conf_tr_k, _ = ovr_kernel.predict_with_confidence(X_train_s)
    pred_val_k, conf_val_k, _ = ovr_kernel.predict_with_confidence(X_val_s)
    pred_test_k, conf_test_k, _ = ovr_kernel.predict_with_confidence(X_test_s)
    acc_tr_k = utils.accuracy(y_train, pred_tr_k)
    acc_val_k = utils.accuracy(y_val, pred_val_k)
    acc_test_k = utils.accuracy(y_test, pred_test_k)
    print(f"Kernel SVM accuracy -> train={acc_tr_k:.3f}  val={acc_val_k:.3f}  test={acc_test_k:.3f}")
    utils.calibration_check(y_test, pred_test_k, conf_test_k, "Kernel/test")
    results["kernel"] = {"train": acc_tr_k, "val": acc_val_k, "test": acc_test_k}
    results["kernel_name"] = kernel_name

    utils.plot_confusion(y_test, pred_test_k, classes, f"{FIGURES_DIR}/kernel_confusion_test.png",
                          f"Kernel SVM ({kernel_name}) - Test confusion matrix")
    utils.plot_linear_vs_kernel(results, f"{FIGURES_DIR}/linear_vs_kernel.png")

    # ---------------- Part E: calibration check (for report) ----------------
    calib_rows = []
    for name, preds, confs in [("linear", pred_test, conf_test), ("kernel", pred_test_k, conf_test_k)]:
        c, w = utils.calibration_check(y_test, preds, confs, f"{name}/test(summary)")
        # NaN isn't valid JSON -- use null when there were no wrong predictions to average
        c = None if np.isnan(c) else c
        w = None if np.isnan(w) else w
        calib_rows.append({"model": name, "mean_conf_correct": c, "mean_conf_wrong": w})
    with open(f"{MODELS_DIR}/calibration_check.json", "w") as f:
        json.dump(calib_rows, f, indent=2)
    with open(f"{MODELS_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---------------- persist everything evaluate.py needs ----------------
    scaler.save(f"{MODELS_DIR}/scaler.npz")
    ovr_linear.save(f"{MODELS_DIR}/linear_ovr_svm.pkl")
    ovr_kernel.save(f"{MODELS_DIR}/kernel_ovr_svm.pkl")
    with open(f"{MODELS_DIR}/classes.json", "w") as f:
        json.dump(classes, f)
    best = "kernel" if acc_test_k >= acc_test else "linear"
    with open(f"{MODELS_DIR}/config.json", "w") as f:
        json.dump({"best_model": best, "kernel_name": kernel_name}, f, indent=2)

    print(f"\nSaved models + scaler to {MODELS_DIR}/, figures to {FIGURES_DIR}/.")
    print(f"Best model on held-out test split: {best} (kernel={acc_test_k:.3f} vs linear={acc_test:.3f})")

    # ---------------- optional: simulated "hidden test set" check ----------------
    # If data/hidden_test_demo/<class>/*.jpg exists (a folder of images the split
    # above never saw -- e.g. a disjoint capture session, or your own phone
    # photos), evaluate on it the same way the instructor's hidden test set
    # described in the assignment will be scored.
    hidden_dir = "data/hidden_test_demo"
    if os.path.isdir(hidden_dir):
        print(f"\n=== Simulated hidden-test check ({hidden_dir}/) ===")
        for name, model in [("linear", ovr_linear), ("kernel", ovr_kernel)]:
            model_for_eval = model
            acc, per_class = utils.evaluate_directory_accuracy(model_for_eval, classes, scaler, hidden_dir)
            if acc is not None:
                print(f"  {name}: hidden-test accuracy = {acc:.3f}  per-class={per_class}")
                with open(f"{MODELS_DIR}/hidden_test_results_{name}.json", "w") as f:
                    json.dump({"accuracy": acc, "per_class": per_class}, f, indent=2)


if __name__ == "__main__":
    main()
