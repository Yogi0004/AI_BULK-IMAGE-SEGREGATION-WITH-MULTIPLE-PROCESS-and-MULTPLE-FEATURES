"""
=============================================================================
  Homes247 — Blur Detection Training Pipeline
  train.py
=============================================================================

PURPOSE
    Trains a MobileNetV2-based regression model that outputs a quality
    score (0–100) for any real estate image.

    Score ≥ threshold  →  Good Quality  (saved to good_quality/)
    Score <  threshold →  Bad  Quality  (saved to bad_quality/)

USAGE
    # Full run (generate + train):
    python train.py

    # Custom source folder:
    python train.py --source "C:/Users/Homes247/Desktop/Bulk_image"

    # Skip dataset generation if already done:
    python train.py --skip_generate

OUTPUTS
    blur_model.h5             ← trained model  (drop into Bulk_image/)
    blur_model_metadata.json  ← accuracy stats
    blur_dataset/             ← synthetic training images + labels.csv

HOW IT WORKS
    1. Takes every image in your source folder
    2. Applies 8 blur levels to each  → 8× dataset automatically
    3. Trains MobileNetV2 in two phases:
         Phase 1: frozen backbone  →  fast convergence
         Phase 2: fine-tune top layers  →  higher accuracy
    4. Evaluates binary accuracy (Good vs Bad) on held-out test split
    5. Saves the best checkpoint throughout training

INTEGRATION INTO app.py  (3-line change)
    from train import assess_image_quality          # ← add this import

    BLUR_MODEL_PATH = os.path.join(BASE_DIR, "blur_model.h5")   # ← config

    # inside process_single_image() — replace old call:
    quality_score, metrics = assess_image_quality(
        image_path,
        category   = category,          # "floorplan"|"masterplan"|"gallery"
        model_path = BLUR_MODEL_PATH,
    )
=============================================================================
"""

import os, csv, json, time, random, argparse, warnings
import numpy as np
from pathlib import Path
from datetime import datetime

import os, csv, json, time, random, argparse, warnings
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Silence TF logs ─────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import cv2

# ── Silence TF logs ──────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import cv2

# ── Lazy TF import so the file can be imported without GPU / TF installed ────
_tf = None
def _get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        _tf = tf
    return _tf


# =============================================================================
#  SECTION 0 — DEFAULTS (edit these to match your machine)
# =============================================================================

BASE_DIR        = r"C:\Users\Homes247\Desktop\Bulk_image"
DATASET_DIR     = r"C:\Users\Homes247\Desktop\blur_dataset"
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "blur_model.h5")

IMG_SIZE        = 224   # MobileNetV2 input size
EPOCHS          = 5
BATCH_SIZE      = 8
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.10
SAMPLES_PER_IMG = 8     # blur levels generated per source image

# Score above this = Good Quality   (matches sidebar default "Good 55")
GOOD_THRESHOLD  = 0.55   # on 0-1 scale  →  55 on 0-100 scale


# =============================================================================
#  SECTION 1 — BLUR LEVEL DEFINITIONS
#  Each entry: (gaussian_kernel, sigma, quality_score_0_to_1, label)
# =============================================================================

BLUR_LEVELS = [
    ( 0,  0.0, 1.00, "PERFECT"),     # sharp original      → 100
    ( 3,  0.8, 0.88, "EXCELLENT"),   # barely noticeable   → 88
    ( 5,  1.2, 0.75, "GOOD"),        # slight softness     → 75
    ( 9,  2.0, 0.58, "ACCEPTABLE"),  # mild blur           → 58
    (13,  3.0, 0.40, "POOR"),        # clearly blurry      → 40
    (19,  5.0, 0.22, "BAD"),         # heavy blur          → 22
    (27,  8.0, 0.10, "VERY_BAD"),    # very heavy          → 10
    (41, 12.0, 0.03, "TERRIBLE"),    # unusable            →  3
]


# =============================================================================
#  SECTION 2 — IMAGE TYPE DETECTION  (no classifier needed)
# =============================================================================

def detect_image_type(gray: np.ndarray) -> str:
    """
    Heuristic — works without the classifier.
    Returns: "floorplan" | "masterplan" | "gallery"
    """
    total     = float(gray.size)
    white_pct = float(np.sum(gray > 210)) / total
    dark_pct  = float(np.sum(gray < 50))  / total
    mid_pct   = 1.0 - white_pct - dark_pct

    if white_pct > 0.55 and dark_pct < 0.10:
        return "floorplan"
    if mid_pct > 0.45 and white_pct < 0.40:
        return "masterplan"
    return "gallery"


# =============================================================================
#  SECTION 3 — OPENCV QUALITY METRICS  (used with or without AI model)
# =============================================================================

def _cv_score_base(gray: np.ndarray) -> tuple:
    """
    Returns (overall_score_0_to_100, metrics_dict)
    Runs on a 512×512 normalised crop.
    """
    g = gray.astype(np.float32)

    # 1. Laplacian variance — primary sharpness measure
    lap_var = float(cv2.Laplacian(g, cv2.CV_32F).var())

    # 2. Tenengrad — gradient energy
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float(np.mean(sx**2 + sy**2))

    # 3. Block sharpness ratio (64×64 blocks, threshold lap>30)
    h, w = g.shape
    BS   = 64
    sharp_blks = sum(
        1 for by in range(0, h, BS) for bx in range(0, w, BS)
        if float(cv2.Laplacian(g[by:by+BS, bx:bx+BS], cv2.CV_32F).var()) > 30
    )
    total_blks  = max(1, (h // BS) * (w // BS))
    block_ratio = sharp_blks / total_blks

    # 4. Canny edge density
    edges       = cv2.Canny(gray.astype(np.uint8), 50, 150)
    edge_density= float(np.sum(edges > 0)) / float(edges.size)

    # 5. Contrast (std-dev)
    contrast = float(np.std(g))

    # Normalise each metric to 0-100
    lap_s   = float(np.clip(lap_var    / 600.0  * 100, 0, 100))
    ten_s   = float(np.clip(tenengrad  / 2500.0 * 100, 0, 100))
    blk_s   = block_ratio * 100.0
    edg_s   = float(np.clip(edge_density / 0.10 * 100, 0, 100))
    con_s   = float(np.clip(contrast   / 55.0   * 100, 0, 100))

    raw = lap_s*0.35 + ten_s*0.25 + blk_s*0.20 + edg_s*0.12 + con_s*0.08

    # Hard caps for clearly blurry images
    if   lap_var  <  5.0:                              raw = min(raw,  5.0)
    elif lap_var  < 15.0 and edge_density < 0.01:      raw = min(raw, 12.0)
    elif lap_var  < 30.0 and tenengrad    < 200.0:     raw = min(raw, 22.0)
    elif lap_var  < 80.0 and block_ratio  < 0.20:      raw = min(raw, 35.0)
    elif lap_var  < 150.0 and block_ratio < 0.35:      raw = min(raw, 48.0)
    if   contrast <  8.0:                              raw = min(raw,  8.0)
    elif contrast < 15.0:                              raw = min(raw, 20.0)

    score = float(np.clip(raw, 0, 100))

    return score, {
        "cv_score":    round(score, 2),
        "lap_var":     round(lap_var,     2),
        "tenengrad":   round(tenengrad,   2),
        "block_ratio": round(block_ratio, 4),
        "edge_density":round(edge_density,4),
        "contrast":    round(contrast,    2),
        "lap_score":   round(lap_s, 2),
        "ten_score":   round(ten_s, 2),
        "blk_score":   round(blk_s, 2),
        "edg_score":   round(edg_s, 2),
        "con_score":   round(con_s, 2),
    }


def _floorplan_cv(gray: np.ndarray) -> float:
    """Sharpness measured ONLY on the drawn lines — white background ignored."""
    u8  = gray.astype(np.uint8)
    _, m = cv2.threshold(u8, 200, 255, cv2.THRESH_BINARY_INV)
    m    = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=2)
    px   = int(np.sum(m > 0))
    if px < 500:
        return 5.0                               # blank / no content
    gf  = gray.astype(np.float32)
    lap = cv2.Laplacian(gf, cv2.CV_32F)
    var = float(np.var(lap[m > 0]))
    s   = float(np.clip(var / 800.0 * 100.0, 0, 100))
    if (px / float(gray.size)) < 0.02:
        s = min(s, 30.0)
    return round(s, 2)


def _masterplan_cv(gray: np.ndarray, bgr: np.ndarray) -> float:
    """Sharpness measured at colour boundary regions."""
    gray_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr is not None else gray.astype(np.uint8)
    edges   = cv2.Canny(gray_u8, 30, 100)
    mask    = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    if int(np.sum(mask > 0)) < 200:
        return 10.0
    gf  = gray.astype(np.float32)
    lap = cv2.Laplacian(gf, cv2.CV_32F)
    var = float(np.var(lap[mask > 0]))
    s   = float(np.clip(var / 1200.0 * 100.0, 0, 100))
    sx  = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
    sy  = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
    ten = float(np.clip(float(np.mean(sx**2 + sy**2)) / 2500.0 * 100.0, 0, 100))
    return round(float(np.clip(s * 0.65 + ten * 0.35, 0, 100)), 2)


def _gallery_cv(gray: np.ndarray) -> float:
    """Laplacian + Tenengrad + FFT high-frequency + texture block variance."""
    gf      = gray.astype(np.float32)
    lap_var = float(cv2.Laplacian(gf, cv2.CV_32F).var())
    lap_s   = float(np.clip(lap_var / 500.0 * 100.0, 0, 100))
    sx      = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
    sy      = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
    ten_s   = float(np.clip(float(np.mean(sx**2 + sy**2)) / 2500.0 * 100.0, 0, 100))
    fft     = np.abs(np.fft.fftshift(np.fft.fft2(gf / 255.0)))
    cy, cx  = gf.shape[0] // 2, gf.shape[1] // 2
    r       = max(1, int(min(gf.shape) * 0.10))
    lf      = np.zeros(gf.shape, dtype=bool)
    lf[cy-r:cy+r, cx-r:cx+r] = True
    hf_r    = 1.0 - float(np.sum(fft[lf])) / (float(np.sum(fft)) + 1e-9)
    fft_s   = float(np.clip((hf_r - 0.60) / (0.95 - 0.60) * 100.0, 0, 100))
    h, w, BS= gf.shape[0], gf.shape[1], 64
    hi_v    = sum(1 for by in range(0, h, BS) for bx in range(0, w, BS)
                  if float(np.var(gf[by:by+BS, bx:bx+BS])) > 150)
    tex_s   = float(hi_v / (((h // BS) * (w // BS)) or 1) * 100.0)
    score   = lap_s * 0.35 + ten_s * 0.30 + fft_s * 0.20 + tex_s * 0.15
    if   lap_var <  5.0: score = min(score,  5.0)
    elif lap_var < 20.0: score = min(score, 20.0)
    elif lap_var < 60.0: score = min(score, 38.0)
    return round(float(np.clip(score, 0, 100)), 2)


# =============================================================================
#  SECTION 4 — INFERENCE  (drop-in replace for assess_image_quality in app.py)
# =============================================================================

_MODEL_CACHE: dict = {}

def _load_model(path: str):
    if path not in _MODEL_CACHE:
        if path and os.path.exists(path):
            tf = _get_tf()
            _MODEL_CACHE[path] = tf.keras.models.load_model(path)
        else:
            _MODEL_CACHE[path] = None
    return _MODEL_CACHE[path]


def assess_image_quality(
    image_path:   str,
    category:     str   = "auto",
    model_path:   str   = None,
    model_weight: float = 0.60,
) -> tuple:
    """
    =========================================================================
    DROP-IN replacement for assess_image_quality() in app.py

    Parameters
    ----------
    image_path   : full path to the image file
    category     : "floorplan" | "masterplan" | "gallery" | "auto"
    model_path   : path to trained blur_model.h5  (None = CV-only mode)
    model_weight : how much to trust AI vs CV  (0.0–0.9, default 0.60)

    Returns
    -------
    (quality_score: float,  metrics: dict)
        quality_score  0–100
        quality_score >= threshold → Good Quality
        quality_score <  threshold → Bad  Quality
    =========================================================================
    """
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return 0.0, {"error": "Cannot read image"}

        # Resize to evaluation size
        EVAL    = 512
        ev_bgr  = cv2.resize(img_bgr, (EVAL, EVAL), interpolation=cv2.INTER_LINEAR)
        ev_gray = cv2.cvtColor(ev_bgr, cv2.COLOR_BGR2GRAY)

        # Auto-detect category if not specified
        if category in (None, "auto", "rejected", "error", ""):
            category = detect_image_type(ev_gray)
        cat = category.lower().strip()

        # ── CV score ────────────────────────────────────────────────────────
        cv_base, cv_m = _cv_score_base(ev_gray)

        if cat == "floorplan":
            cat_score = _floorplan_cv(ev_gray)
            cv_final  = cat_score * 0.70 + cv_base * 0.30
        elif cat == "masterplan":
            cat_score = _masterplan_cv(ev_gray, ev_bgr)
            cv_final  = cat_score * 0.65 + cv_base * 0.35
        else:  # gallery
            cat_score = _gallery_cv(ev_gray)
            cv_final  = cat_score * 0.60 + cv_base * 0.40

        cv_final = float(np.clip(cv_final, 0, 100))

        # ── AI model score ──────────────────────────────────────────────────
        ai_score = None
        if model_path:
            model = _load_model(model_path)
            if model is not None:
                try:
                    tf      = _get_tf()
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    img_224 = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
                    arr     = tf.keras.applications.mobilenet_v2.preprocess_input(
                                  img_224.astype(np.float32))
                    arr     = np.expand_dims(arr, 0)
                    raw     = float(model.predict(arr, verbose=0)[0][0])
                    ai_score= round(raw * 100.0, 2)
                except Exception:
                    ai_score = None

        # ── Blend ───────────────────────────────────────────────────────────
        if ai_score is not None:
            w           = float(np.clip(model_weight, 0.0, 0.9))
            final_score = ai_score * w + cv_final * (1.0 - w)
        else:
            final_score = cv_final

        # ── Absolute hard caps (protect against edge cases) ─────────────────
        lv, ed, ct = cv_m["lap_var"], cv_m["edge_density"], cv_m["contrast"]
        if   lv <  3.0:               final_score = min(final_score,  4.0)
        elif lv < 10.0 and ed < 0.005:final_score = min(final_score, 10.0)
        elif lv < 25.0 and ed < 0.01: final_score = min(final_score, 20.0)
        if   ct <  5.0:               final_score = min(final_score,  6.0)

        final_score = round(float(np.clip(final_score, 0, 100)), 2)

        # ── Build metrics dict (compatible with existing app.py keys) ────────
        return final_score, {
            # Keys already used by app.py
            "overall":       final_score,
            "sharpness":     cv_m["lap_score"],
            "tenengrad":     cv_m["ten_score"],
            "block_content": cv_m["blk_score"],
            "edge_visible":  cv_m["edg_score"],
            "contrast":      cv_m["con_score"],
            "blur":          cv_m["lap_score"],
            "edge":          cv_m["edg_score"],
            "brightness":    cv_m["con_score"],
            # Extra debug keys
            "cv_score":      round(cv_final,   2),
            "cat_score":     round(cat_score,  2),
            "ai_score":      ai_score,
            "category_used": cat,
            "laplacian_raw": cv_m["lap_var"],
            "edge_density":  cv_m["edge_density"],
            "contrast_raw":  cv_m["contrast"],
            "block_ratio":   cv_m["block_ratio"],
        }

    except Exception as e:
        return 0.0, {"error": f"assess_image_quality failed: {e}"}


# =============================================================================
#  SECTION 5 — DATASET GENERATION
# =============================================================================

def _apply_blur(img: np.ndarray, kernel: int, sigma: float) -> np.ndarray:
    if kernel == 0:
        return img.copy()
    k = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.GaussianBlur(img, (k, k), sigma)


def _augment(img: np.ndarray) -> np.ndarray:
    """Light augmentation applied to the sharp version before blurring."""
    factor = random.uniform(0.85, 1.15)
    img    = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    if random.random() > 0.5:
        q   = random.randint(60, 95)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def generate_dataset(source_dir: str, output_dir: str) -> str:
    """
    Scans source_dir for images.
    Creates SAMPLES_PER_IMG blurred copies of each.
    Saves images to output_dir/images/ and labels to output_dir/labels.csv.
    Returns path to labels.csv.
    """
    imgs_dir    = os.path.join(output_dir, "images")
    labels_path = os.path.join(output_dir, "labels.csv")
    os.makedirs(imgs_dir, exist_ok=True)

    exts  = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = [p for p in Path(source_dir).rglob("*") if p.suffix.lower() in exts]

    if not paths:
        raise FileNotFoundError(
            f"No images found in {source_dir}\n"
            "Check the path and ensure images exist."
        )

    print(f"\n  Source images : {len(paths)}")
    print(f"  Total samples : {len(paths) * SAMPLES_PER_IMG}")

    rows       = []
    skipped    = 0
    start      = time.time()

    for idx, p in enumerate(paths):
        # Progress display
        pct = (idx + 1) / len(paths) * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        eta = ((time.time() - start) / (idx + 1)) * (len(paths) - idx - 1)
        print(f"  [{bar}] {pct:5.1f}%  ETA {int(eta//60)}m{int(eta%60):02d}s", end="\r")

        img = cv2.imread(str(p))
        if img is None:
            skipped += 1
            continue

        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_type = detect_image_type(gray)
        img      = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
        img_aug  = _augment(img)

        for lvl_idx, (ks, sigma, score, label) in enumerate(BLUR_LEVELS[:SAMPLES_PER_IMG]):
            blurred  = _apply_blur(img_aug, ks, sigma)
            out_name = f"img{idx:05d}_b{lvl_idx}_{label}.jpg"
            cv2.imwrite(
                os.path.join(imgs_dir, out_name),
                blurred,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            rows.append({
                "filename":   out_name,
                "score":      score,
                "label":      label,
                "img_type":   img_type,
                "blur_level": lvl_idx,
                "is_good":    1 if score >= GOOD_THRESHOLD else 0,
            })

    print()  # newline after progress bar

    with open(labels_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    good = sum(1 for r in rows if r["is_good"] == 1)
    bad  = len(rows) - good
    print(f"  ✅ Dataset ready — {len(rows)} samples  "
          f"({good} good / {bad} bad)  skipped={skipped}")
    print(f"  Labels: {labels_path}")
    return labels_path


# =============================================================================
#  SECTION 6 — TF.DATA PIPELINE
# =============================================================================

def _build_tf_dataset(filenames, scores, images_dir, batch, augment=False, shuffle=False):
    tf = _get_tf()
    sep = os.sep

    def load(fn, sc):
        path = tf.strings.join([images_dir + sep, fn])
        raw  = tf.io.read_file(path)
        img  = tf.image.decode_jpeg(raw, channels=3)
        img  = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img  = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img, tf.reshape(tf.cast(sc, tf.float32), (1,))

    def aug_fn(img, sc):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.12)
        img = tf.image.random_contrast(img, 0.88, 1.12)
        img = tf.image.random_saturation(img, 0.88, 1.12)
        return img, sc

    ds = tf.data.Dataset.from_tensor_slices((filenames, scores))
    if shuffle:
        ds = ds.shuffle(len(filenames), reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# =============================================================================
#  SECTION 7 — MODEL ARCHITECTURE
# =============================================================================

def _build_model():
    tf = _get_tf()
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import MobileNetV2

    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False   # frozen for Phase 1

    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x   = base(inp, training=False)

    x   = layers.Dense(256)(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Activation("relu")(x)
    x   = layers.Dropout(0.35)(x)

    x   = layers.Dense(128)(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Activation("relu")(x)
    x   = layers.Dropout(0.25)(x)

    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.Dropout(0.15)(x)

    out = layers.Dense(1, activation="sigmoid", name="quality_score")(x)

    return Model(inp, out, name="BlurDetector"), base


def _compile(model, lr: float, steps: int):
    tf = _get_tf()
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.optimizers.schedules import CosineDecay
    schedule = CosineDecay(lr, decay_steps=steps, alpha=1e-6)
    model.compile(
        optimizer=Adam(schedule),
        loss="mse",
        metrics=[
            "mae",
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )


# =============================================================================
#  SECTION 8 — TRAINING (two-phase)
# =============================================================================

def train_model(labels_path: str, model_save: str, epochs: int, batch_size: int):
    """
    Two-phase training:
      Phase 1 — head only (backbone frozen)  ~epochs/3
      Phase 2 — fine-tune top 40% of backbone

    Returns (model, metadata_dict)
    """
    tf = _get_tf()
    from tensorflow.keras import callbacks

    print(f"\n  Labels : {labels_path}")

    # ── Load + split ─────────────────────────────────────────────────────────
    with open(labels_path) as f:
        rows = list(csv.DictReader(f))
    random.shuffle(rows)

    n       = len(rows)
    n_test  = max(1, int(n * TEST_SPLIT))
    n_val   = max(1, int(n * VAL_SPLIT))
    n_train = n - n_val - n_test

    tr = rows[:n_train]
    va = rows[n_train:n_train + n_val]
    te = rows[n_train + n_val:]
    images_dir = os.path.join(os.path.dirname(labels_path), "images")

    def to_lists(r):
        return [x["filename"] for x in r], [float(x["score"]) for x in r]

    tr_f, tr_s = to_lists(tr)
    va_f, va_s = to_lists(va)
    te_f, te_s = to_lists(te)

    print(f"  Split  : train={n_train}  val={n_val}  test={n_test}")

    train_ds = _build_tf_dataset(tr_f, tr_s, images_dir, batch_size, augment=True,  shuffle=True)
    val_ds   = _build_tf_dataset(va_f, va_s, images_dir, batch_size, augment=False, shuffle=False)
    test_ds  = _build_tf_dataset(te_f, te_s, images_dir, batch_size, augment=False, shuffle=False)

    steps_per_epoch = max(1, n_train // batch_size)
    os.makedirs(os.path.dirname(model_save) or ".", exist_ok=True)
    best_ckpt = model_save.replace(".h5", "_best.h5")

    def make_callbacks(patience):
        return [
            callbacks.ModelCheckpoint(best_ckpt, monitor="val_loss",
                                       save_best_only=True, verbose=1),
            callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                    restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4,
                                        patience=3, min_lr=1e-7, verbose=1),
            callbacks.CSVLogger(model_save.replace(".h5", "_history.csv")),
        ]

    model, base = _build_model()

    # ── Phase 1: train head only ──────────────────────────────────────────────
    p1_epochs = max(5, epochs // 3)
    print(f"\n{'─'*60}")
    print(f"  Phase 1 — Head only  ({p1_epochs} epochs, lr=1e-3)")
    print(f"{'─'*60}\n")
    _compile(model, lr=1e-3, steps=steps_per_epoch * p1_epochs)
    model.fit(train_ds, validation_data=val_ds, epochs=p1_epochs,
              callbacks=make_callbacks(patience=7), verbose=1)

    # ── Phase 2: fine-tune top 40% of backbone ────────────────────────────────
    p2_epochs   = epochs - p1_epochs
    n_layers    = len(base.layers)
    freeze_till = int(n_layers * 0.60)
    base.trainable = True
    for layer in base.layers[:freeze_till]:
        layer.trainable = False

    print(f"\n{'─'*60}")
    print(f"  Phase 2 — Fine-tune top {n_layers - freeze_till} layers  "
          f"({p2_epochs} epochs, lr=2e-5)")
    print(f"{'─'*60}\n")
    _compile(model, lr=2e-5, steps=steps_per_epoch * p2_epochs)
    model.fit(train_ds, validation_data=val_ds, epochs=p2_epochs,
              callbacks=make_callbacks(patience=5), verbose=1)

    # ── Evaluation ────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Evaluating on test set …")
    results  = model.evaluate(test_ds, verbose=0)
    mse, mae, rmse = results[0], results[1], results[2]

    preds    = model.predict(test_ds, verbose=0).flatten()
    actuals  = np.array(te_s)
    pb       = (preds   >= GOOD_THRESHOLD).astype(int)
    ab       = (actuals >= GOOD_THRESHOLD).astype(int)
    accuracy = float(np.mean(pb == ab) * 100)

    tp = int(np.sum((pb==1) & (ab==1)))
    tn = int(np.sum((pb==0) & (ab==0)))
    fp = int(np.sum((pb==1) & (ab==0)))
    fn = int(np.sum((pb==0) & (ab==1)))
    precision = tp / (tp + fp + 1e-9) * 100
    recall    = tp / (tp + fn + 1e-9) * 100
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    print(f"\n  ── Results ──────────────────────────────────────────")
    print(f"     MSE       : {mse:.5f}")
    print(f"     MAE       : {mae:.5f}  (avg score error ±{mae*100:.1f} pts)")
    print(f"     RMSE      : {rmse:.5f}")
    print(f"     Accuracy  : {accuracy:.2f}%  (Good vs Bad)")
    print(f"     Precision : {precision:.2f}%")
    print(f"     Recall    : {recall:.2f}%")
    print(f"     F1 Score  : {f1:.2f}%")
    print(f"\n  ── Confusion Matrix ─────────────────────────────────")
    print(f"              Pred Good  Pred Bad")
    print(f"  True Good    {tp:6d}    {fn:6d}")
    print(f"  True Bad     {fp:6d}    {tn:6d}")

    # ── Save final model ──────────────────────────────────────────────────────
    model.save(model_save)
    print(f"\n  ✅ Model saved  → {model_save}")

    metadata = {
        "trained_at":          datetime.now().isoformat(),
        "total_samples":       n,
        "train_samples":       n_train,
        "val_samples":         n_val,
        "test_samples":        n_test,
        "epochs_phase1":       p1_epochs,
        "epochs_phase2":       p2_epochs,
        "batch_size":          batch_size,
        "img_size":            IMG_SIZE,
        "good_threshold":      GOOD_THRESHOLD,
        "test_mse":            round(float(mse),  6),
        "test_mae":            round(float(mae),  6),
        "test_rmse":           round(float(rmse), 6),
        "test_accuracy_pct":   round(accuracy,    2),
        "test_precision_pct":  round(precision,   2),
        "test_recall_pct":     round(recall,       2),
        "test_f1_pct":         round(f1,           2),
        "confusion":           {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }
    meta_path = model_save.replace(".h5", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  📋 Metadata     → {meta_path}")

    return model, metadata


# =============================================================================
#  SECTION 9 — MAIN CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
        description="Homes247 Blur Detection — Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py
  python train.py --source "C:/Users/Homes247/Desktop/Bulk_image"
  python train.py --source ./images --epochs 30 --batch 16
  python train.py --skip_generate   # re-train without regenerating dataset
        """,
    )
    parser.add_argument("--source",        default=BASE_DIR,         help="Source images folder")
    parser.add_argument("--dataset",       default=DATASET_DIR,      help="Dataset output folder")
    parser.add_argument("--model",         default=MODEL_SAVE_PATH,  help="Model save path (.h5)")
    parser.add_argument("--epochs",        type=int, default=EPOCHS, help="Total training epochs")
    parser.add_argument("--batch",         type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--skip_generate", action="store_true",      help="Skip dataset generation")
    args = parser.parse_args()

    print("=" * 60)
    print("  Homes247 — Blur Detection Training Pipeline")
    print("=" * 60)
    print(f"  Source  : {args.source}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Model   : {args.model}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")

    t0 = time.time()

    # ── Step 1: Generate dataset ───────────────────────────────────────────
    labels_path = os.path.join(args.dataset, "labels.csv")
    if args.skip_generate and os.path.exists(labels_path):
        print(f"\n  ⏭  Skipping generation (using {labels_path})")
    else:
        print(f"\n{'─'*60}")
        print(f"  STEP 1/2 — Generating Dataset")
        print(f"{'─'*60}")
        labels_path = generate_dataset(args.source, args.dataset)

    # ── Step 2: Train ──────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  STEP 2/2 — Training Model")
    print(f"{'─'*60}")
    _, meta = train_model(labels_path, args.model, args.epochs, args.batch)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  ✅  DONE — {int(elapsed//60)}m {int(elapsed%60)}s")
    print(f"  Accuracy  : {meta['test_accuracy_pct']:.2f}%")
    print(f"  F1 Score  : {meta['test_f1_pct']:.2f}%")
    print(f"  Model     : {args.model}")
    print(f"{'='*60}")
    print()
    print("  NEXT STEPS TO USE IN app.py:")
    print("  ─────────────────────────────────────────────────────")
    print("  1. Add at top of app.py:")
    print("       from train import assess_image_quality")
    print()
    print("  2. Add in CONFIG section of app.py:")
    print("       BLUR_MODEL_PATH = os.path.join(BASE_DIR, 'blur_model.h5')")
    print()
    print("  3. In process_single_image(), change the call to:")
    print("       quality_score, metrics = assess_image_quality(")
    print("           image_path,")
    print("           category   = category,")
    print("           model_path = BLUR_MODEL_PATH,")
    print("       )")
    print("  ─────────────────────────────────────────────────────")