#!/usr/bin/env python3
"""
evaluate.py
-----------
Required deliverable: a single command-line script that accepts an image
path and prints the predicted fruit name and confidence score.

Usage:
    python evaluate.py path/to/photo.jpg
    python evaluate.py path/to/photo.jpg --model linear
    python evaluate.py path/to/dir_of_photos/            # batch mode
    python evaluate.py path/to/photo.jpg --topk 3         # show top-k classes

Loads the already-trained model from models/ (see train.py) -- no
retraining happens here.
"""

import argparse
import os
import sys
import numpy as np

from features import extract_features_from_path
import utils


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def predict_one(path, model, classes, scaler, topk=1):
    feats = extract_features_from_path(path)
    feats_s = scaler.transform(feats.reshape(1, -1))
    pred_classes, confidences, proba = model.predict_with_confidence(feats_s)
    pred_idx = pred_classes[0]
    conf = confidences[0]
    order = np.argsort(-proba[0])[:topk]
    topk_list = [(classes[model.classes_[i]], float(proba[0, i])) for i in order]
    return classes[pred_idx], float(conf), topk_list


def main():
    parser = argparse.ArgumentParser(description="Predict fruit class + confidence for an image.")
    parser.add_argument("path", help="Path to an image file, or a directory of images (batch mode).")
    parser.add_argument("--model", choices=["linear", "kernel"], default=None,
                         help="Which trained model to use. Defaults to the best model chosen at "
                              "train time (see models/config.json).")
    parser.add_argument("--topk", type=int, default=1, help="Show top-k class probabilities.")
    args = parser.parse_args()

    model, classes, scaler, model_name = utils.load_trained_model(args.model)
    print(f"[using {model_name} SVM model, classes={classes}]\n")

    if os.path.isdir(args.path):
        files = sorted(f for f in os.listdir(args.path)
                        if os.path.splitext(f)[1].lower() in IMG_EXTS)
        if not files:
            print(f"No image files found in {args.path}")
            sys.exit(1)
        for fname in files:
            full = os.path.join(args.path, fname)
            try:
                pred, conf, topk_list = predict_one(full, model, classes, scaler, args.topk)
            except Exception as e:
                print(f"{fname}: ERROR ({e})")
                continue
            topk_str = ", ".join(f"{c}={p*100:.1f}%" for c, p in topk_list)
            print(f"{fname:30s} -> {pred:12s} confidence={conf*100:5.1f}%   [{topk_str}]")
    else:
        pred, conf, topk_list = predict_one(args.path, model, classes, scaler, args.topk)
        print(f"Predicted fruit : {pred}")
        print(f"Confidence      : {conf*100:.1f}%")
        if args.topk > 1:
            print("Top-{}:".format(args.topk))
            for c, p in topk_list:
                print(f"  {c:12s} {p*100:5.1f}%")


if __name__ == "__main__":
    main()
