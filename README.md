# Fruit Image Classification with SVM & Kernels

A from-scratch multi-class SVM pipeline (Module 06/07 assignment): hand-crafted
image features -> One-vs-Rest linear SVM -> kernelized (RBF) SVM via kernel-Pegasos
-> Platt-scaled confidence scores.

**No `sklearn.svm`, no `sklearn.multiclass`, no deep learning.** Only `numpy` for the
math; `opencv-python` / `PIL` for image I/O and classical feature extraction.

## The dataset

`data/raw/<class>/` contains **real photographs** from the
[Fruits-360](https://github.com/Horea94/Fruit-Images-Dataset) dataset (pulled directly
from its GitHub source), covering 6 classes:

| Class | Images | Varieties mixed in |
|---|---|---|
| apple | 150 | Braeburn, Golden, Granny Smith, Red, Red Delicious |
| banana | 150 | Banana, Lady Finger, Red |
| orange | 150 | Orange |
| mango | 150 | Mango, Mango Red |
| grape | 150 | Blue, Pink, White |
| strawberry | 150 | Strawberry, Strawberry Wedge |

All 900 images are real photos, well above the assignment's 80-images-per-class
minimum, and each class mixes multiple varieties/colors (e.g. red, green, and
yellow apples) rather than a single lookalike variety, for realistic intra-class
diversity.

`data/hidden_test_demo/<class>/` holds 40 images per class pulled from Fruits-360's
separate `Test/` folder — a different capture session, never touched by
`preprocess.py`'s train/val/test split — used to simulate the assignment's
"instructor's hidden test set" protocol (see `models/hidden_test_results_*.json`
and `report.pdf` §6 for the results and an important honesty note about what this
does and doesn't prove).

## Pipeline / how to run

```bash
pip install -r requirements.txt

# Parts A, C, D, E in one go: cleans + splits data/raw/ 70/15/15, caches Part-B
# feature vectors, trains linear + kernel OvR SVMs, calibrates confidence,
# saves models/ + models/figures/*.png + models/results.json
python train.py

# Required deliverable: predict a single image (or a whole folder)
python evaluate.py path/to/photo.jpg --topk 3
python evaluate.py path/to/folder_of_photos/
```

## Project layout

```
fruit_svm_project/
  data/
    raw/<class>/            # 150 real Fruits-360 photos per class (Part A input)
    processed/                # cached feature arrays + train/val/test split (from train.py)
    hidden_test_demo/<class>/  # 40 real photos/class from a disjoint capture session
  models/
    figures/                    # training curves, confusion matrices, comparison chart
    *.pkl, *.npz, *.json          # trained models, scaler, metrics
  features.py                      # Part B: color / shape / texture feature extraction
  kernels.py                        # Part D: linear / polynomial / RBF kernel functions
  svm.py                              # Parts C/D/E: LinearSVM, KernelSVM (kernel-Pegasos),
                                       #   PlattScaler, OneVsRestSVM -- all from scratch
  train.py                             # Part A preprocessing + training + evaluation, all-in-one
  evaluate.py                           # REQUIRED: CLI, image -> (class, confidence)
  utils.py                               # shared: data split/cleaning, StandardScaler,
                                          #   metrics, plotting, model loading
  requirements.txt
  README.md
  report.pdf                              # written report
```

## What's implemented from scratch (per Technical Requirements)

- **Hinge-loss objective** and **mini-batch sub-gradient descent** training loop — `svm.py::LinearSVM`
- **Kernel functions** `K(x,z)` (linear, polynomial, RBF), vectorized — `kernels.py`
- **Kernelized training** via the kernel-Pegasos algorithm (no QP solver needed) — `svm.py::KernelSVM`
- **One-vs-Rest decision logic** (train K binary SVMs, `argmax_k f_k(x)` at predict time) — `svm.py::OneVsRestSVM`
- **Confidence calibration** (1-D Platt/sigmoid scaling fit per class on the validation split, then
  renormalized across classes) — `svm.py::PlattScaler`

## Results

See `models/results.json`, `models/calibration_check.json`, `models/hidden_test_results_*.json`,
and the plots in `models/figures/` for the actual numbers this pipeline produced — `report.pdf`
walks through them in full, **including an honest discussion of why Fruits-360's clean studio
conditions let both models reach ~100% and what that does and doesn't tell you about
real-world generalization.**
