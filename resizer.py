"""
================================================================
Homes247 — AI-Powered Blur Detection System
================================================================
Supports: Floor Plan, Master Plan, Gallery images
Output  : quality_score 0-100 (pure blur/sharpness, no resolution bias)
Author  : Production module for Homes247 v2.5
================================================================
"""

import os
import cv2
import numpy as np
from pathlib import Path

# ── Optional deep-learning imports (graceful fallback) ───────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms, models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════════
# 1.  CLASSIC CV METRICS  (fast, always-on baseline)
# ════════════════════════════════════════════════════════════════════════════

def _cv_blur_score(gray_eval: np.ndarray) -> dict:
    """
    All-in-one OpenCV blur metrics computed on a normalised 512×512 float32 array.
    Returns individual component scores AND a combined cv_score (0-100).
    """
    h, w = gray_eval.shape
    gf   = gray_eval.astype(np.float32)

    # 1a. Laplacian variance — primary sharpness indicator
    lap_var  = float(cv2.Laplacian(gf, cv2.CV_32F).var())

    # 1b. Tenengrad — edge gradient energy
    sx       = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
    sy       = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad= float(np.mean(sx**2 + sy**2))

    # 1c. Block sharpness ratio — % of image blocks that have visible detail
    BS = 64   # block size
    sharp_blocks = sum(
        1 for by in range(0, h, BS) for bx in range(0, w, BS)
        if float(cv2.Laplacian(
            gf[by:by+BS, bx:bx+BS], cv2.CV_32F).var()) > 30
    )
    total_blocks = ((h // BS) * (w // BS)) or 1
    block_ratio  = sharp_blocks / total_blocks

    # 1d. Edge density via Canny
    u8           = gray_eval.astype(np.uint8)
    edges        = cv2.Canny(u8, 50, 150)
    edge_density = float(np.sum(edges > 0)) / float(edges.size)

    # 1e. Contrast (std-dev of pixel values)
    contrast = float(np.std(gf))

    # ── Scale each metric to 0-100 ──────────────────────────────────────────
    lap_s  = float(np.clip(lap_var     / 600.0  * 100, 0, 100))
    ten_s  = float(np.clip(tenengrad   / 2500.0 * 100, 0, 100))
    blk_s  = block_ratio * 100.0
    edg_s  = float(np.clip(edge_density/ 0.10   * 100, 0, 100))
    con_s  = float(np.clip(contrast    / 55.0   * 100, 0, 100))

    raw = lap_s*0.35 + ten_s*0.25 + blk_s*0.20 + edg_s*0.12 + con_s*0.08

    # ── Hard blur penalty caps ──────────────────────────────────────────────
    if   lap_var  <  5.0:                          raw = min(raw,  5.0)
    elif lap_var  < 15.0 and edge_density < 0.01:  raw = min(raw, 12.0)
    elif lap_var  < 30.0 and tenengrad    < 200.0: raw = min(raw, 22.0)
    elif lap_var  < 80.0 and block_ratio  < 0.20:  raw = min(raw, 35.0)
    elif lap_var  <150.0 and block_ratio  < 0.35:  raw = min(raw, 48.0)
    if   contrast <  8.0:                          raw = min(raw,  8.0)
    elif contrast < 15.0:                          raw = min(raw, 20.0)

    return {
        "cv_score":    float(np.clip(raw, 0, 100)),
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


# ════════════════════════════════════════════════════════════════════════════
# 2.  CATEGORY-AWARE CV SCORING
#     Floor plans have large white areas — standard metrics fail them.
#     We detect the image TYPE and adjust thresholds.
# ════════════════════════════════════════════════════════════════════════════

def _detect_image_type(gray: np.ndarray) -> str:
    """
    Heuristic to detect floor-plan vs master-plan vs gallery WITHOUT
    relying on the classifier model — used as fallback inside quality check.
    """
    total      = gray.size
    white_pct  = float(np.sum(gray > 210)) / total   # very bright pixels
    dark_pct   = float(np.sum(gray < 50))  / total   # very dark pixels
    mid_pct    = 1.0 - white_pct - dark_pct

    # Floor plans: mostly white + thin dark lines
    if white_pct > 0.55 and dark_pct < 0.10:
        return "floorplan"

    # Master plans: colourful, lots of mid-tones
    img_bgr = None  # not available here; mid_pct proxy
    if mid_pct > 0.45 and white_pct < 0.40:
        return "masterplan"

    return "gallery"


def _floorplan_blur_score(gray_eval: np.ndarray) -> float:
    """
    Floor plans are mostly white with thin lines.
    Standard Laplacian gets confused by the huge white areas.
    Strategy: focus ONLY on the line/detail regions.
    """
    u8     = gray_eval.astype(np.uint8)

    # Isolate the lines (dark pixels on white background)
    _, line_mask = cv2.threshold(u8, 200, 255, cv2.THRESH_BINARY_INV)

    # Morphological dilation to give lines a neighbourhood
    kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    line_area = cv2.dilate(line_mask, kernel, iterations=2)

    line_pixels = int(np.sum(line_area > 0))
    if line_pixels < 500:
        # Virtually blank page → very bad
        return 5.0

    # Measure sharpness ONLY inside line regions
    gf       = gray_eval.astype(np.float32)
    lap      = cv2.Laplacian(gf, cv2.CV_32F)
    lap_line = lap[line_area > 0]

    line_var = float(np.var(lap_line))

    # Thin-line sharpness reference:  sharp > 800, blurry < 80
    score = float(np.clip(line_var / 800.0 * 100.0, 0, 100))

    # Extra penalty: if lines cover <5% of image they may be absent/blurry
    coverage = line_pixels / float(gray_eval.size)
    if coverage < 0.02:
        score = min(score, 30.0)

    return round(score, 2)


def _masterplan_blur_score(gray_eval: np.ndarray, bgr_eval: np.ndarray) -> float:
    """
    Master plans have colour regions + text + boundaries.
    We measure sharpness at colour boundaries (region edges).
    """
    gf  = gray_eval.astype(np.float32)
    u8  = gray_eval.astype(np.uint8)

    # Find region boundaries using Canny on the colour image
    if bgr_eval is not None:
        gray_c = cv2.cvtColor(bgr_eval, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    else:
        gray_c = u8

    edges     = cv2.Canny(gray_c, 30, 100)
    edge_mask = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    n_edge = int(np.sum(edge_mask > 0))
    if n_edge < 200:
        return 10.0

    lap       = cv2.Laplacian(gf, cv2.CV_32F)
    lap_edges = lap[edge_mask > 0]
    edge_var  = float(np.var(lap_edges))

    # Reference: sharp boundary > 1200, blurry < 100
    score = float(np.clip(edge_var / 1200.0 * 100.0, 0, 100))

    # Tenengrad bonus
    sx  = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
    sy  = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
    ten = float(np.mean(sx**2 + sy**2))
    ten_s = float(np.clip(ten / 2500.0 * 100.0, 0, 100))

    combined = score * 0.65 + ten_s * 0.35
    return round(combined, 2)


def _gallery_blur_score(gray_eval: np.ndarray, bgr_eval: np.ndarray) -> float:
    """
    Gallery photos — standard approach works well but we add FFT
    high-frequency energy analysis on top for robustness.
    """
    gf = gray_eval.astype(np.float32)

    # Laplacian
    lap_var = float(cv2.Laplacian(gf, cv2.CV_32F).var())
    lap_s   = float(np.clip(lap_var / 500.0 * 100.0, 0, 100))

    # Tenengrad
    sx  = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
    sy  = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
    ten = float(np.mean(sx**2 + sy**2))
    ten_s = float(np.clip(ten / 2500.0 * 100.0, 0, 100))

    # FFT high-frequency ratio
    fft      = np.abs(np.fft.fftshift(np.fft.fft2(gf / 255.0)))
    cy, cx   = gf.shape[0]//2, gf.shape[1]//2
    r        = max(1, int(min(gf.shape) * 0.10))
    lf_mask  = np.zeros(gf.shape, dtype=bool)
    lf_mask[cy-r:cy+r, cx-r:cx+r] = True
    hf_ratio = 1.0 - float(np.sum(fft[lf_mask])) / (float(np.sum(fft)) + 1e-9)
    fft_s    = float(np.clip((hf_ratio - 0.60) / (0.95 - 0.60) * 100.0, 0, 100))

    # Local block variance in texture regions
    h, w     = gf.shape
    BS       = 64
    high_var = sum(
        1 for by in range(0, h, BS) for bx in range(0, w, BS)
        if float(np.var(gf[by:by+BS, bx:bx+BS])) > 150
    )
    total    = ((h//BS)*(w//BS)) or 1
    tex_s    = float(high_var / total * 100.0)

    score = lap_s*0.35 + ten_s*0.30 + fft_s*0.20 + tex_s*0.15

    # Hard penalty
    if   lap_var <  5.0: score = min(score,  5.0)
    elif lap_var < 20.0: score = min(score, 20.0)
    elif lap_var < 60.0: score = min(score, 38.0)

    return round(float(np.clip(score, 0, 100)), 2)


# ════════════════════════════════════════════════════════════════════════════
# 3.  PYTORCH BLUR DETECTION MODEL
#     Lightweight CNN that outputs a single blur score (0-1).
#     Trained on synthetic blur data (see training section below).
# ════════════════════════════════════════════════════════════════════════════

class BlurDetectorCNN(nn.Module):
    """
    Lightweight MobileNetV3-Small backbone fine-tuned for blur regression.
    Input : (B, 3, 224, 224) normalised image
    Output: (B, 1) blur score in [0, 1]  →  multiply by 100 for display
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        backbone       = models.mobilenet_v3_small(pretrained=pretrained)
        # Replace final classifier
        in_features    = backbone.classifier[3].in_features
        backbone.classifier[3] = nn.Identity()
        self.backbone  = backbone
        self.head      = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()          # output in [0, 1]
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


# ════════════════════════════════════════════════════════════════════════════
# 4.  TENSORFLOW / KERAS BLUR DETECTION MODEL
#     Alternative for setups that already use TF (like your classifier.h5)
# ════════════════════════════════════════════════════════════════════════════

def build_tf_blur_model(input_shape=(224, 224, 3)):
    """
    MobileNetV2-based blur regressor.
    Fine-tune from ImageNet weights then replace head with regression.
    """
    base   = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False,
        weights='imagenet', pooling='avg'
    )
    base.trainable = False   # freeze during initial training

    inp    = tf.keras.Input(shape=input_shape)
    x      = base(inp, training=False)
    x      = layers.Dropout(0.3)(x)
    x      = layers.Dense(128, activation='relu')(x)
    x      = layers.Dropout(0.2)(x)
    out    = layers.Dense(1, activation='sigmoid')(x)   # 0-1

    model  = Model(inp, out, name="BlurDetectorTF")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='mse',
        metrics=['mae']
    )
    return model


# ════════════════════════════════════════════════════════════════════════════
# 5.  SYNTHETIC TRAINING DATA GENERATOR
#     Creates blur dataset from your existing images automatically.
#     Run this ONCE to generate training data, then train the model.
# ════════════════════════════════════════════════════════════════════════════

def generate_blur_dataset(
    source_dir: str,
    output_dir: str,
    samples_per_image: int = 8
):
    """
    Takes clean sharp images from source_dir.
    Applies varying levels of Gaussian blur.
    Saves images + labels to output_dir.

    Dataset structure created:
        output_dir/
            images/
                img_001_blur0.jpg   ← original sharp (label=1.0)
                img_001_blur1.jpg   ← light blur     (label=0.80)
                img_001_blur2.jpg   ← medium blur    (label=0.55)
                img_001_blur3.jpg   ← heavy blur     (label=0.25)
                ...
            labels.csv              ← filename, score columns
    """
    import csv

    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    labels_path = os.path.join(output_dir, "labels.csv")

    # Blur levels: (kernel_size, sigma) → ground-truth score
    BLUR_LEVELS = [
        (0,    0.0,  1.00),   # no blur       → perfect
        (3,    0.5,  0.90),   # very slight   → excellent
        (5,    1.0,  0.78),   # slight        → good
        (7,    1.5,  0.65),   # mild          → acceptable
        (11,   2.5,  0.50),   # moderate      → borderline
        (15,   4.0,  0.35),   # strong        → bad
        (21,   6.0,  0.18),   # very strong   → very bad
        (31,  10.0,  0.05),   # extreme       → terrible
    ]

    exts   = {'.jpg', '.jpeg', '.png', '.webp'}
    images = [p for p in Path(source_dir).rglob('*') if p.suffix.lower() in exts]

    rows   = []
    count  = 0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Resize to model input size
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)

        for (ks, sigma, score) in BLUR_LEVELS[:samples_per_image]:
            if ks == 0:
                blurred = img.copy()
            else:
                blurred = cv2.GaussianBlur(img, (ks, ks), sigma)

            out_name = f"{img_path.stem}_{count:06d}_b{ks}.jpg"
            out_path = os.path.join(output_dir, "images", out_name)
            cv2.imwrite(out_path, blurred, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append({"filename": out_name, "score": score})
            count += 1

    with open(labels_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "score"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dataset generated: {count} images → {output_dir}")
    return labels_path


# ════════════════════════════════════════════════════════════════════════════
# 6.  PYTORCH TRAINING PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def train_pytorch_blur_model(
    dataset_dir: str,
    save_path:   str   = "blur_detector.pth",
    epochs:      int   = 20,
    batch_size:  int   = 32,
    lr:          float = 1e-4,
):
    """
    Full training loop for the PyTorch blur detector.
    Run once on your server/PC, then use the saved .pth for inference.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not installed. pip install torch torchvision")

    import csv
    from torch.utils.data import Dataset, DataLoader, random_split

    # ── Dataset ────────────────────────────────────────────────────────────
    class BlurDataset(Dataset):
        def __init__(self, root, labels_csv, transform):
            self.root      = root
            self.transform = transform
            with open(labels_csv) as f:
                self.rows = list(csv.DictReader(f))

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            row   = self.rows[idx]
            path  = os.path.join(self.root, "images", row["filename"])
            img   = cv2.imread(path)
            img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            score = float(row["score"])
            if self.transform:
                from PIL import Image as PILImage
                img = PILImage.fromarray(img)
                img = self.transform(img)
            return img, torch.tensor([score], dtype=torch.float32)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    labels_csv = os.path.join(dataset_dir, "labels.csv")
    full_ds    = BlurDataset(dataset_dir, labels_csv, transform)
    n_val      = max(1, int(len(full_ds) * 0.15))
    n_train    = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    # ── Model ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = BlurDetectorCNN(pretrained=True).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn= nn.MSELoss()

    best_val = float('inf')

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        tr_loss = 0.0
        for imgs, scores in train_dl:
            imgs, scores = imgs.to(device), scores.to(device)
            opt.zero_grad()
            preds = model(imgs)
            loss  = loss_fn(preds, scores)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(imgs)
        tr_loss /= n_train

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, scores in val_dl:
                imgs, scores = imgs.to(device), scores.to(device)
                preds = model(imgs)
                val_loss += loss_fn(preds, scores).item() * len(imgs)
        val_loss /= n_val
        sched.step()

        print(f"Epoch {epoch:3d}/{epochs}  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Saved best model → {save_path}")

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    return save_path


# ════════════════════════════════════════════════════════════════════════════
# 7.  TENSORFLOW TRAINING PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def train_tf_blur_model(
    dataset_dir: str,
    save_path:   str  = "blur_detector_tf.h5",
    epochs:      int  = 20,
    batch_size:  int  = 32,
):
    """
    Full TensorFlow training loop.
    Preferred if you want ONE framework (you already use TF for classifier.h5)
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not installed.")

    import csv
    import tensorflow as tf

    labels_csv  = os.path.join(dataset_dir, "labels.csv")
    images_dir  = os.path.join(dataset_dir, "images")

    with open(labels_csv) as f:
        rows = list(csv.DictReader(f))

    filenames = [os.path.join(images_dir, r["filename"]) for r in rows]
    scores    = [float(r["score"]) for r in rows]

    # ── tf.data pipeline ───────────────────────────────────────────────────
    def load_and_preprocess(path, score):
        raw   = tf.io.read_file(path)
        img   = tf.image.decode_jpeg(raw, channels=3)
        img   = tf.image.resize(img, [224, 224])
        img   = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img, score

    def augment(img, score):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        return img, score

    dataset   = tf.data.Dataset.from_tensor_slices((filenames, scores))
    dataset   = dataset.shuffle(len(filenames))
    n_val     = max(1, int(len(filenames) * 0.15))
    val_ds    = dataset.take(n_val).map(load_and_preprocess).batch(batch_size).prefetch(2)
    train_ds  = dataset.skip(n_val).map(load_and_preprocess).map(augment).batch(batch_size).prefetch(2)

    model = build_tf_blur_model()
    model.summary()

    cb = [
        tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=True,
                                           monitor='val_loss', verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                             patience=3, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6,
                                         restore_best_weights=True),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cb)
    model.save(save_path)
    print(f"\n✅ Model saved → {save_path}")
    return save_path


# ════════════════════════════════════════════════════════════════════════════
# 8.  MODEL LOADER  (singleton, cached)
# ════════════════════════════════════════════════════════════════════════════

_blur_model_cache = {}

def load_blur_model(model_path: str, backend: str = "auto"):
    """
    Loads and caches the trained blur model.
    backend: "torch" | "tf" | "auto" (tries torch first, then tf)
    Returns None if no model found (falls back to CV-only mode).
    """
    global _blur_model_cache

    key = (model_path, backend)
    if key in _blur_model_cache:
        return _blur_model_cache[key]

    if not os.path.exists(model_path):
        return None

    ext = Path(model_path).suffix.lower()

    # ── PyTorch ────────────────────────────────────────────────────────────
    if (backend in ("torch", "auto")) and TORCH_AVAILABLE and ext == ".pth":
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            m      = BlurDetectorCNN(pretrained=False)
            m.load_state_dict(torch.load(model_path, map_location=device))
            m.to(device)
            m.eval()
            _blur_model_cache[key] = ("torch", m, device)
            print(f"✅ Loaded PyTorch blur model from {model_path}")
            return _blur_model_cache[key]
        except Exception as e:
            print(f"⚠️  Could not load PyTorch model: {e}")

    # ── TensorFlow ─────────────────────────────────────────────────────────
    if (backend in ("tf", "auto")) and TF_AVAILABLE and ext in (".h5", ".keras"):
        try:
            m = tf.keras.models.load_model(model_path)
            _blur_model_cache[key] = ("tf", m, None)
            print(f"✅ Loaded TF blur model from {model_path}")
            return _blur_model_cache[key]
        except Exception as e:
            print(f"⚠️  Could not load TF model: {e}")

    return None


def _infer_with_model(model_tuple, image_path: str) -> float:
    """
    Run inference with loaded model. Returns score 0-100.
    """
    backend, model, device = model_tuple

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return -1.0

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_224 = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    if backend == "torch":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(img_224)
        tensor  = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            score = float(model(tensor).squeeze().cpu()) * 100.0
        return round(score, 2)

    elif backend == "tf":
        arr     = tf.keras.applications.mobilenet_v2.preprocess_input(
                      img_224.astype(np.float32))
        arr     = np.expand_dims(arr, 0)
        score   = float(model.predict(arr, verbose=0)[0][0]) * 100.0
        return round(score, 2)

    return -1.0


# ════════════════════════════════════════════════════════════════════════════
# 9.  MAIN PUBLIC API  ← this replaces assess_image_quality() in your app.py
# ════════════════════════════════════════════════════════════════════════════

def assess_image_quality(
    image_path:  str,
    category:    str  = "auto",      # "floorplan" | "masterplan" | "gallery" | "auto"
    model_path:  str  = None,        # path to .pth or .h5 blur model (optional)
    model_weight: float = 0.55,      # how much to trust AI model vs CV (0-1)
) -> tuple:
    """
    ════════════════════════════════════════════════════════
    DROP-IN REPLACEMENT for the existing assess_image_quality()
    in your Homes247 app.py.

    Returns
    -------
    (quality_score: float,  metrics: dict)

    quality_score : 0-100
        0-10  = Completely blurry / blank
        10-25 = Very blurry
        25-40 = Blurry
        40-55 = Acceptable
        55-70 = Clear
        70-85 = Sharp
        85-100= Crystal clear

    metrics : dict with detailed component scores
    ════════════════════════════════════════════════════════
    """
    try:
        # ── Load image ──────────────────────────────────────────────────────
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return 0, {"error": "Cannot read image"}

        # ── Resize to evaluation canvas (eliminates resolution bias) ────────
        EVAL = 512
        img_eval  = cv2.resize(img_bgr, (EVAL, EVAL),
                               interpolation=cv2.INTER_LINEAR)
        gray_eval = cv2.cvtColor(img_eval, cv2.COLOR_BGR2GRAY)

        # ── Auto-detect category if not provided ────────────────────────────
        if category == "auto":
            gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            category  = _detect_image_type(gray_full)

        cat = category.lower().strip()

        # ══════════════════════════════════════════════════════════════════
        # STEP A: Category-aware CV score
        # ══════════════════════════════════════════════════════════════════
        cv_metrics = _cv_blur_score(gray_eval)
        cv_base    = cv_metrics["cv_score"]

        if cat == "floorplan":
            cat_score = _floorplan_blur_score(gray_eval)
            # Blend: floor-plan-specific (70%) + generic CV (30%)
            cv_final  = cat_score * 0.70 + cv_base * 0.30

        elif cat == "masterplan":
            cat_score = _masterplan_blur_score(gray_eval, img_eval)
            cv_final  = cat_score * 0.65 + cv_base * 0.35

        else:  # gallery
            cat_score = _gallery_blur_score(gray_eval, img_eval)
            cv_final  = cat_score * 0.60 + cv_base * 0.40

        cv_final = float(np.clip(cv_final, 0, 100))

        # ══════════════════════════════════════════════════════════════════
        # STEP B: AI model score (if model is available)
        # ══════════════════════════════════════════════════════════════════
        ai_score   = None
        model_info = None

        if model_path:
            model_tuple = load_blur_model(model_path)
            if model_tuple is not None:
                raw_ai = _infer_with_model(model_tuple, image_path)
                if raw_ai >= 0:
                    ai_score   = raw_ai
                    model_info = model_tuple[0]   # "torch" or "tf"

        # ══════════════════════════════════════════════════════════════════
        # STEP C: Combine CV + AI
        # ══════════════════════════════════════════════════════════════════
        if ai_score is not None:
            w_ai = float(np.clip(model_weight, 0.0, 0.9))
            w_cv = 1.0 - w_ai
            final_score = ai_score * w_ai + cv_final * w_cv
        else:
            final_score = cv_final

        final_score = round(float(np.clip(final_score, 0, 100)), 2)

        # ══════════════════════════════════════════════════════════════════
        # STEP D: Absolute hard caps (catches edge cases both CV & AI miss)
        # ══════════════════════════════════════════════════════════════════
        lv = cv_metrics["lap_var"]
        ed = cv_metrics["edge_density"]
        ct = cv_metrics["contrast"]

        if   lv  <  3.0:              final_score = min(final_score,  4.0)
        elif lv  < 10.0 and ed < 0.005: final_score = min(final_score, 10.0)
        elif lv  < 25.0 and ed < 0.01:  final_score = min(final_score, 20.0)
        if   ct  <  5.0:              final_score = min(final_score,  6.0)

        final_score = round(float(np.clip(final_score, 0, 100)), 2)

        # ── Build metrics dict (backward-compatible with existing app.py) ──
        metrics = {
            # ── Primary output ──────────────────────────────
            "overall":        final_score,
            "cv_score":       round(cv_final,  2),
            "cat_score":      round(cat_score, 2),
            "ai_score":       round(ai_score,  2) if ai_score is not None else None,
            "ai_backend":     model_info,
            "category_used":  cat,

            # ── Backward-compatible keys used by app.py ─────
            "sharpness":      round(cv_metrics["lap_score"],  2),
            "tenengrad":      round(cv_metrics["ten_score"],  2),
            "block_content":  round(cv_metrics["blk_score"],  2),
            "edge_visible":   round(cv_metrics["edg_score"],  2),
            "contrast":       round(cv_metrics["con_score"],  2),
            "blur":           round(cv_metrics["lap_score"],  2),
            "edge":           round(cv_metrics["edg_score"],  2),
            "brightness":     round(cv_metrics["con_score"],  2),

            # ── Raw values for debugging ─────────────────────
            "laplacian_raw":  cv_metrics["lap_var"],
            "tenengrad_raw":  cv_metrics["tenengrad"],
            "block_ratio":    cv_metrics["block_ratio"],
            "edge_density":   cv_metrics["edge_density"],
            "contrast_raw":   cv_metrics["contrast"],
        }

        return final_score, metrics

    except Exception as e:
        return 0, {"error": f"Quality assessment failed: {str(e)}"}