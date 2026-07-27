"""
features.py
-----------
Hand-crafted feature extraction for classical (non-deep-learning) fruit
classification (Part B). We deliberately do NOT feed raw pixels into the
SVM -- a 128x128 RGB image is 49,152 raw numbers, mostly redundant and
highly sensitive to translation/background. Instead we build a compact
feature vector combining three complementary signal types:

  1. COLOR   - HSV channel histograms. Fruits are very often separable by
               color alone (bananas yellow, oranges orange, grapes purple/
               green), and HSV separates color (Hue) from lighting
               brightness (Value) better than RGB, which matters a lot
               once the hidden test set has different lighting than ours.
  2. SHAPE   - after a simple foreground/background segmentation, we
               measure the fruit silhouette's aspect ratio, extent
               (area / bounding-box area), and circularity
               (4*pi*Area/Perimeter^2). This is what distinguishes a
               round apple from an elongated banana from a bunch of
               grapes, independent of color.
  3. TEXTURE - Sobel-based edge density and a simple Histogram of
               Oriented Gradients (HOG) descriptor, which captures
               surface texture (smooth mango skin vs. bumpy strawberry
               seeds vs. the granular skin of an orange).

All three are concatenated into one fixed-length vector per image.
"""

import numpy as np
import cv2


IMG_SIZE = 128  # every image is resized to IMG_SIZE x IMG_SIZE before features are computed

# ----- feature-vector layout (documented for the report) --------------------
N_COLOR_BINS = 16        # per channel
N_COLOR_FEATURES = N_COLOR_BINS * 3       # H, S, V histograms concatenated
N_SHAPE_FEATURES = 4                       # aspect_ratio, extent, circularity, area_frac
N_HOG_CELLS = 8 * 8                        # 8x8 grid of cells
N_HOG_BINS = 9
N_TEXTURE_FEATURES = N_HOG_CELLS * N_HOG_BINS + 1   # + global edge density
FEATURE_LENGTH = N_COLOR_FEATURES + N_SHAPE_FEATURES + N_TEXTURE_FEATURES


def load_and_resize(path, size=IMG_SIZE):
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def segment_foreground(img_bgr):
    """
    Simple, fast foreground/background split: Otsu threshold on the
    saturation channel (fruit is usually saturated, background/table/
    plate is usually closer to gray/white), cleaned up with morphology.
    Good enough for this classical-features pipeline -- not meant to be
    a general-purpose segmenter.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    _, mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def color_features(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    feats = []
    ranges = [(0, 180), (0, 256), (0, 256)]  # H in [0,180), S,V in [0,256)
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [N_COLOR_BINS], ranges[ch])
        hist = hist.flatten() / (hist.sum() + 1e-8)
        feats.append(hist)
    return np.concatenate(feats)


def shape_features(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    if not contours:
        return np.zeros(N_SHAPE_FEATURES)
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    x, y, bw, bh = cv2.boundingRect(c)
    aspect_ratio = bw / (bh + 1e-8)
    extent = area / (bw * bh + 1e-8)
    circularity = (4 * np.pi * area) / (perim ** 2 + 1e-8)
    area_frac = area / (h * w)
    return np.array([aspect_ratio, extent, circularity, area_frac])


def texture_features(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = (np.arctan2(gy, gx) * 180 / np.pi) % 180  # unsigned gradient angle [0,180)

    edge_density = np.mean(mag > (0.2 * mag.max() + 1e-8))

    # simple HOG: split image into an 8x8 grid of cells, histogram gradient
    # orientations (weighted by magnitude) per cell into 9 bins
    h, w = gray.shape
    n_cells_side = 8
    cell_h, cell_w = h // n_cells_side, w // n_cells_side
    hog_feats = []
    bin_edges = np.linspace(0, 180, N_HOG_BINS + 1)
    for i in range(n_cells_side):
        for j in range(n_cells_side):
            cell_mag = mag[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            cell_ang = ang[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            hist, _ = np.histogram(cell_ang, bins=bin_edges, weights=cell_mag)
            hist = hist / (hist.sum() + 1e-8)
            hog_feats.append(hist)
    hog_feats = np.concatenate(hog_feats)
    return np.concatenate([hog_feats, [edge_density]])


def extract_features(img_bgr):
    """Full Part-B feature vector for a single (already resized) BGR image."""
    mask = segment_foreground(img_bgr)
    feats = np.concatenate([
        color_features(img_bgr),
        shape_features(mask),
        texture_features(img_bgr),
    ])
    return feats.astype(np.float64)


def extract_features_from_path(path):
    img = load_and_resize(path)
    return extract_features(img)


if __name__ == "__main__":
    print(f"Feature vector length: {FEATURE_LENGTH} "
          f"(color={N_COLOR_FEATURES}, shape={N_SHAPE_FEATURES}, texture={N_TEXTURE_FEATURES})")
