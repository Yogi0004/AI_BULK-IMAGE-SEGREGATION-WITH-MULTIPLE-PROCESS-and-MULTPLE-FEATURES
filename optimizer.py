from __future__ import annotations
from turtle import pd
"""
Image Optimizer Module — Complete Image Processing Pipeline
============================================================
Extracts ALL processing logic from Bulk_image/app.py into a standalone module
that can be imported by main.py without Streamlit conflicts.

Features: AI classification, quality assessment, text/logo removal (OCR),
auto-resize, watermarking, session reports, ZIP creation, server export,
statistics tracking, and settings management.
"""

import os
import sys
import json
import time
import uuid
import shutil
import hashlib
import zipfile
import urllib.request
import urllib.parse
import traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
import streamlit as st

# Load .env file for credentials (R2 keys, etc.)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass  # dotenv not installed — env vars must be set externally

# Heavy deps — lazy loaded to avoid ImportError if missing
np: Any    = None
cv2: Any   = None
Image: Any = None
torch: Any = None
nn: Any    = None
F: Any     = None

def _ensure_deps():
    """Lazy-import heavy dependencies on first use."""
    global np, cv2, Image, torch, nn, F
    if np is None:
        import numpy
        np = numpy
    if cv2 is None:
        import cv2 as _cv2
        cv2 = _cv2
    if Image is None:
        from PIL import Image as _Image
        Image = _Image
    if torch is None:
        import torch as _torch
        import torch.nn as _nn
        import torch.nn.functional as _f
        torch, nn, F = _torch, _nn, _f


# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Homes247 Premium - Image Processing",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== MODEL LOADING (TF Classifier) =====================
@st.cache_resource
def load_ai_model():
    BASE_DIR = r"C:\Users\Homes247\Desktop\Bulk_image"
    MODEL_PATH = os.path.join(BASE_DIR, 'classifier.h5')
    try:
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH)
        CLASSES = ['floorplan', 'gallery', 'masterplan']
        return model, CLASSES, BASE_DIR
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, ['floorplan', 'gallery', 'masterplan'], None

model, CLASSES, BASE_DIR = load_ai_model()
if model is None:
    st.error("❌ CRITICAL: AI model failed to load from classifier.h5 — check the path.")

# ===================== OCR LOADING =====================
@st.cache_resource(show_spinner="Loading AI OCR engine for text removal...")
def load_ocr_reader():
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        return reader
    except ImportError:
        st.warning("⚠️ EasyOCR not installed. Text removal will be skipped.")
        return None
    except Exception as e:
        st.warning(f"⚠️ OCR loading failed: {e}. Text removal will be skipped.")
        return None

ocr_reader = load_ocr_reader()

# ===================== PYTORCH QUALITY MODEL =====================
_qual_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blur_model = models.mobilenet_v2(pretrained=True)
blur_model.classifier = torch.nn.Identity()
blur_model = blur_model.to(_qual_device)
blur_model.eval()

transform_blur = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ===================== RRDB ARCHITECTURE (Real-ESRGAN backbone) =====================

def make_layer(block, n_layers):
    return nn.Sequential(*[block() for _ in range(n_layers)])


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(nf,        gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc,   gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + gc*2, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + gc*3, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + gc*4, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc)
        self.RDB2 = ResidualDenseBlock(nf, gc)
        self.RDB3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        return self.RDB3(self.RDB2(self.RDB1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """Real-ESRGAN RRDB network. num_block=23 for x4, num_block=6 for anime."""
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64,
                 num_block=23, num_grow_ch=32, scale=4):
        super().__init__()
        self.scale       = scale
        self.conv_first  = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body        = make_layer(lambda: RRDB(num_feat, num_grow_ch), num_block)
        self.conv_body   = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1    = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2    = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if scale == 8:
            self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr     = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last   = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu       = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body = self.conv_body(self.body(feat))
        feat = feat + body
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        if self.scale == 8:
            feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode='nearest')))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


# ===================== SR MODEL REGISTRY =====================
SR_MODELS = {
    "🏠 Real-ESRGAN ×4  (Best for Photos)": {
        "url":       "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename":  "RealESRGAN_x4plus.pth",
        "num_block": 23,
        "scale":     4,
    },
    "🏢 Real-ESRGAN ×2  (Gentle Upscale)": {
        "url":       "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "filename":  "RealESRGAN_x2plus.pth",
        "num_block": 23,
        "scale":     2,
    },
    "🖼️  Real-ESRGAN ×4 Anime  (Renders/CG)": {
        "url":       "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "filename":  "RealESRGAN_x4plus_anime_6B.pth",
        "num_block": 6,
        "scale":     4,
    },
}

WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)


# ===================== SR HELPER FUNCTIONS =====================

@st.cache_resource(show_spinner=False)
def get_sr_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def download_sr_weights(url: str, dest: Path, prog):
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return
    def _hook(count, block_size, total_size):
        if total_size > 0:
            prog.progress(min(count * block_size / total_size, 1.0),
                          text="Downloading model…")
    urllib.request.urlretrieve(url, dest, reporthook=_hook)


@st.cache_resource(show_spinner=False)
def load_sr_model(model_key):
    info   = SR_MODELS[model_key]
    sr_dev = get_sr_device()
    dest   = WEIGHTS_DIR / info["filename"]

    prog = st.progress(0.0, text="Downloading model weights…")
    download_sr_weights(info["url"], dest, prog)
    prog.empty()

    state = torch.load(dest, map_location="cpu")
    payload = None
    for key in ("params_ema", "params", "state_dict"):
        if key in state:
            payload = state[key]
            break
    if payload is None:
        payload = state

    remapped = {}
    for k, v in payload.items():
        new_k = k.replace(".rdb1.", ".RDB1.").replace(".rdb2.", ".RDB2.").replace(".rdb3.", ".RDB3.")
        remapped[new_k] = v

    sr_model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64,
        num_block=info["num_block"],
        num_grow_ch=32,
        scale=info["scale"]
    )

    missing, _ = sr_model.load_state_dict(remapped, strict=False)
    critical = [k for k in missing if "conv" in k]
    if critical:
        raise RuntimeError(f"Model load failed — critical keys missing: {critical[:3]}")

    torch.set_grad_enabled(False)
    sr_model.eval().to(sr_dev)
    return sr_model, info["scale"], sr_dev


def pil_to_tensor(img: Image.Image, sr_dev) -> torch.Tensor:
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(sr_dev)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).permute(1, 2, 0).cpu().float().clamp(0.0, 1.0).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def tile_inference(sr_model, lr_t: torch.Tensor, scale: int,
                   tile: int = 256, overlap: int = 32) -> torch.Tensor:
    b, c, h, w = lr_t.shape
    sr_dev  = lr_t.device
    stride  = tile - overlap
    h_idx   = list(range(0, max(1, h - tile), stride)) + [max(0, h - tile)]
    w_idx   = list(range(0, max(1, w - tile), stride)) + [max(0, w - tile)]

    out_h, out_w = h * scale, w * scale
    E = torch.zeros(b, c, out_h, out_w, device=sr_dev)
    W = torch.zeros_like(E)

    out_tile = tile * scale
    cy    = torch.hann_window(out_tile, periodic=False, device=sr_dev)
    cx    = torch.hann_window(out_tile, periodic=False, device=sr_dev)
    blend = (cy.unsqueeze(1) * cx.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    total = len(h_idx) * len(w_idx)
    done  = 0
    bar   = st.progress(0, text=f"Processing tile 1 / {total}…")

    with torch.inference_mode():
        for hi in h_idx:
            for wi in w_idx:
                patch  = lr_t[:, :, hi:hi + tile, wi:wi + tile]
                out    = sr_model(patch).clamp(0, 1)
                oH, oW = out.shape[2], out.shape[3]
                bm     = blend[:, :, :oH, :oW]
                oh, ow = hi * scale, wi * scale
                E[:, :, oh:oh + oH, ow:ow + oW] += out * bm
                W[:, :, oh:oh + oH, ow:ow + oW] += bm
                done += 1
                bar.progress(done / total, text=f"Processing tile {done} / {total}…")

    bar.empty()
    return (E / W.clamp(min=1e-8)).clamp(0, 1)


def run_sr(sr_model, scale: int, sr_dev, img: Image.Image,
           tile: int, overlap: int) -> Image.Image:
    lr_t = pil_to_tensor(img, sr_dev)
    h, w = lr_t.shape[2], lr_t.shape[3]
    with torch.inference_mode():
        if h <= tile and w <= tile:
            sr_t = sr_model(lr_t).clamp(0, 1)
        else:
            sr_t = tile_inference(sr_model, lr_t, scale, tile, overlap)
    return tensor_to_pil(sr_t)


def image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ===================== TEXT REMOVAL KEYWORDS =====================
TEXT_REMOVAL_KEYWORDS = {
    "Master Plan": [
        'master', 'plan', 'legend', 'www', '.com', 'road', 'phase',
        'building', 'block', 'tower', 'future', 'extension', 'services',
        'north', 'entry', 'exit', 'copyright', 'reserved', 'logo',
        'trademark', 'developer', 'architect', 'scale', 'disclaimer',
    ],
    "Floor Plan": [
        'floor', 'plan', 'www', '.com', 'legend', 'scale', 'north',
        'copyright', 'reserved', 'logo', 'trademark', 'developer',
        'architect', 'disclaimer', 'note', 'not to scale',
    ],
    "Gallery": [
        'www', '.com', 'watermark', 'copyright', 'reserved', 'logo',
        'trademark', 'photo', 'image', 'stock', 'getty', 'shutterstock',
        'preview', 'sample', 'draft',
    ],
}

TEXT_REMOVAL_SETTINGS = {
    "Master Plan": {"corner_pct": 0.18, "edge_pct": 0.10, "ocr_margin": 0.22},
    "Floor Plan":  {"corner_pct": 0.15, "edge_pct": 0.08, "ocr_margin": 0.18},
    "Gallery":     {"corner_pct": 0.01, "edge_pct": 0.01, "ocr_margin": 0.1},
}

# ===================== CONFIGURATION =====================
if BASE_DIR:
    OUTPUT_DIR          = os.path.join(BASE_DIR, "api_output")
    UPLOAD_DIR          = os.path.join(OUTPUT_DIR, "uploads")
    TEMP_DIR            = os.path.join(BASE_DIR, "api_temp")
    STATS_FILE          = os.path.join(BASE_DIR, "api_processing_statistics.json")
    UPLOAD_HISTORY_FILE = os.path.join(BASE_DIR, "upload_history.json")
    WATERMARK_LOGO_FILE = os.path.join(BASE_DIR, "watermark_logo.png")
    URL_DOWNLOAD_DIR    = os.path.join(BASE_DIR, "url_downloads")
    SESSION_REPORTS_DIR = os.path.join(BASE_DIR, "session_reports")
    ALL_SESSIONS_FILE   = os.path.join(BASE_DIR, "all_sessions_log.json")
    SETTINGS_FILE       = os.path.join(BASE_DIR, "app_settings.json")
else:
    OUTPUT_DIR          = "./api_output"
    UPLOAD_DIR          = os.path.join(OUTPUT_DIR, "uploads")
    TEMP_DIR            = "./api_temp"
    STATS_FILE          = "./api_processing_statistics.json"
    UPLOAD_HISTORY_FILE = "./upload_history.json"
    WATERMARK_LOGO_FILE = "./watermark_logo.png"
    URL_DOWNLOAD_DIR    = "./url_downloads"
    SESSION_REPORTS_DIR = "./session_reports"
    ALL_SESSIONS_FILE   = "./all_sessions_log.json"
    SETTINGS_FILE       = "./app_settings.json"

CATEGORY_MAPPING = {
    "floorplan":  "Floor Plan Images",
    "masterplan": "Master Plan Images",
    "gallery":    "Gallery Images",
    "rejected":   "Others"
}

CATEGORY_SIZES = {
    "floorplan":  (1500, 1500),
    "masterplan": (1640, 860),
    "gallery":    (820, 430),
    "rejected":   None
}

CATEGORY_TO_TEXT_REMOVAL = {
    "floorplan":  "Floor Plan",
    "masterplan": "Master Plan",
    "gallery":    "Gallery",
    "rejected":   None
}

for _d in [OUTPUT_DIR, UPLOAD_DIR, TEMP_DIR, URL_DOWNLOAD_DIR, SESSION_REPORTS_DIR]:
    os.makedirs(_d, exist_ok=True)
for cat in ["floorplan", "masterplan", "gallery", "rejected"]:
    os.makedirs(os.path.join(OUTPUT_DIR, cat, "good_quality"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, cat, "bad_quality"),  exist_ok=True)

# ===================== SETTINGS =====================
def load_app_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    # ── CHANGED: replaced quality_threshold_label with blur_threshold ─────────
    return {
        "output_format":       "WEBP",
        "confidence_threshold": 45,
        "blur_threshold":       500,
        "sr_model_key":         "🏠 Real-ESRGAN ×4  (Best for Photos)",
        "enable_batch_sr":      False,
    }

def save_app_settings(settings):
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except:
        return False

def get_format_extension(fmt):
    mapping = {
        "WEBP": ("webp", "WEBP", [cv2.IMWRITE_WEBP_QUALITY,    100]),
        "JPEG": ("jpg",  "JPEG", [cv2.IMWRITE_JPEG_QUALITY,    100]),
        "JPG":  ("jpg",  "JPEG", [cv2.IMWRITE_JPEG_QUALITY,    100]),
        "PNG":  ("png",  "PNG",  [cv2.IMWRITE_PNG_COMPRESSION,   0]),
        "AVIF": ("jpg",  "JPEG", [cv2.IMWRITE_JPEG_QUALITY,    100]),
    }
    return mapping.get(fmt, ("webp", "WEBP", [cv2.IMWRITE_WEBP_QUALITY, 100]))

def format_file_size(size_bytes):
    if size_bytes is None:
        return "N/A"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"



# ===================== IMAGE QUALITY — BLUR ONLY (Laplacian Variance) =========
#
# REPLACED: detect_image_type / assess_photo_quality / assess_plan_quality
# NEW:      Single blur-only check, size-normalised to 500×500 so score is NOT
#           affected by image dimensions.
#
# Calibrated from real property images:
#   Sharp floor plan  →  ~2800   (GOOD)
#   Blurred overlay   →  ~200–300 (BAD)
#   Recommended threshold: 500
#
def assess_image_quality(image_path):
    """
    Quality check using ONLY Laplacian Variance (blur detection).
    Always resizes to 500×500 first — score is NOT affected by image size.
    Returns (score, metrics) — drop-in replacement for the old function.
    """
    try:
        pil_img = Image.open(image_path).convert('RGB')
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # Normalise to 500×500 so large images don't get an unfair advantage
        std     = cv2.resize(gray, (500, 500))
        score   = float(cv2.Laplacian(std, cv2.CV_64F).var())
        return score, {
            "sharpness":  round(score, 2),
            "brightness": 0,
            "contrast":   0,
            "blur":       round(max(0.0, 100.0 - min(score / 10.0, 100.0)), 2),
            "edge":       0,
        }
    except Exception:
        return 50.0, {
            "sharpness": 0, "brightness": 0,
            "contrast":  0, "blur":       0, "edge": 0,
        }
# =============================================================================

# ===================== TEXT REMOVAL FUNCTIONS =====================
def build_removal_mask(img_bgr, category, ocr_reader, extra_margin=2, protect_center=True):
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    _sf = max(1.0, max(h, w) / 1000.0)

    cfg      = TEXT_REMOVAL_SETTINGS.get(category, TEXT_REMOVAL_SETTINGS["Gallery"])
    keywords = TEXT_REMOVAL_KEYWORDS.get(category, TEXT_REMOVAL_KEYWORDS["Gallery"])
    ocr_margin = cfg["ocr_margin"] + extra_margin * 0.08

    # ── OCR-based text detection ──────────────────────────────────────────────
    if ocr_reader is not None:
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            results = ocr_reader.readtext(gray)
            _ocr_pad = int(8 * _sf)
            for det in results:
                bbox, text = det[0], det[1]
                pts = np.array(bbox, dtype=np.int32)
                cx = np.mean(pts[:, 0])
                cy = np.mean(pts[:, 1])
                at_edge = (cx < w * ocr_margin or cx > w * (1 - ocr_margin) or
                           cy < h * ocr_margin or cy > h * (1 - ocr_margin))
                if category == "Gallery":
                    text_lower = text.lower()
                    has_keyword = any(kw in text_lower for kw in keywords)
                    should_remove = at_edge and has_keyword
                else:
                    should_remove = at_edge
                if should_remove:
                    exp = pts.copy()
                    exp[:, 0] = np.clip(
                        pts[:, 0] + np.array([-_ocr_pad,  _ocr_pad, _ocr_pad, -_ocr_pad]), 0, w)
                    exp[:, 1] = np.clip(
                        pts[:, 1] + np.array([-_ocr_pad, -_ocr_pad, _ocr_pad,  _ocr_pad]), 0, h)
                    cv2.fillPoly(mask, [exp.astype(np.int32)], 255)
        except Exception:
            pass

    # ── Corner and edge strip masks ───────────────────────────────────────────
    cx_pct = cfg["corner_pct"] + extra_margin * 0.03
    ey_pct = cfg["edge_pct"]   + extra_margin * 0.03
    cxs = int(w * cx_pct)
    cys = int(h * cx_pct)
    es  = int(min(h, w) * ey_pct)
    mask[0:cys, 0:cxs]     = 255
    mask[0:cys, w-cxs:w]   = 255
    mask[h-cys:h, 0:cxs]   = 255
    mask[h-cys:h, w-cxs:w] = 255
    if category != "Gallery":
        mask[0:es, :]   = 255
        mask[h-es:h, :] = 255
        mask[:, 0:es]   = 255
        mask[:, w-es:w] = 255

    # ── Contour-based text/logo detection ─────────────────────────────────────
    gray2 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray2, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = h * w
    edge_check = 0.18

    for cnt in contours:
        area          = 0.0
        x             = 0
        y             = 0
        cw_c          = 1
        ch_c          = 1
        near_edge     = False
        is_small      = False
        is_text_shape = False
        aspect_c      = 1.0
        try:
            area      = cv2.contourArea(cnt)
            x, y, cw_c, ch_c = cv2.boundingRect(cnt)
            near_edge     = (x < w * edge_check or x + cw_c > w * (1 - edge_check) or
                             y < h * edge_check or y + ch_c > h * (1 - edge_check))
            is_small      = area < total_area * 0.03
            aspect_c      = max(cw_c, ch_c) / (min(cw_c, ch_c) + 1)
            is_text_shape = aspect_c > 2.5
        except Exception:
            continue
        if near_edge and (is_small or is_text_shape):
            xp = max(int(5 * _sf), int(cw_c * 0.08))
            yp = max(int(5 * _sf), int(ch_c * 0.08))
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            cv2.rectangle(mask,
                          (max(0, x - xp),        max(0, y - yp)),
                          (min(w, x + cw_c + xp), min(h, y + ch_c + yp)),
                          255, -1)

    # ── Legend detection for plan images ──────────────────────────────────────
    if category in ("Master Plan", "Floor Plan"):
        edges_det = cv2.Canny(gray2, 50, 150)
        _k_sz  = max(3, int(18 * _sf))
        kr     = cv2.getStructuringElement(cv2.MORPH_RECT, (_k_sz, _k_sz))
        closed = cv2.morphologyEx(edges_det, cv2.MORPH_CLOSE, kr)
        rc, _  = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in rc:
            x_l       = 0
            y_l       = 0
            cw_l      = 1
            ch_l      = 1
            a_l       = 0
            is_legend = False
            at_edge_l = False
            asp_l     = 1.0
            try:
                x_l, y_l, cw_l, ch_l = cv2.boundingRect(cnt)
                a_l       = cw_l * ch_l
                is_legend = 0.003 < (a_l / total_area) < 0.12
                at_edge_l = (x_l < w * 0.20 or x_l + cw_l > w * 0.80 or
                             y_l < h * 0.20 or y_l + ch_l > h * 0.80)
                asp_l     = max(cw_l, ch_l) / (min(cw_l, ch_l) + 1)
            except Exception:
                continue
            if is_legend and at_edge_l and 1.0 < asp_l < 4.0:
                _leg_pad = int(8 * _sf)
                cv2.rectangle(mask,
                              (max(0, x_l - _leg_pad),        max(0, y_l - _leg_pad)),
                              (min(w, x_l + cw_l + _leg_pad), min(h, y_l + ch_l + _leg_pad)),
                              255, -1)

        main_area = find_main_plan_area(img_bgr)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(main_area))

    # ── Final mask cleanup ────────────────────────────────────────────────────
    _d_sz = max(2, int(2 * _sf))
    k     = np.ones((_d_sz, _d_sz), np.uint8)
    mask  = cv2.dilate(mask, k, iterations=1)
    _g_sz = max(3, int(3 * _sf)) | 1
    mask  = cv2.GaussianBlur(mask, (_g_sz, _g_sz), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def find_main_plan_area(img_bgr):
    h, w  = img_bgr.shape[:2]
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            if cv2.contourArea(cnt) > h * w * 0.10:
                cv2.drawContours(main_mask, [cnt], -1, 255, -1)
    if np.sum(main_mask) < h * w * 0.2:
        cv2.rectangle(main_mask,
                      (int(w * 0.15), int(h * 0.15)),
                      (int(w * 0.85), int(h * 0.85)), 255, -1)
    kp = np.ones((15, 15), np.uint8)
    main_mask = cv2.dilate(main_mask, kp, iterations=2)
    return main_mask


def remove_text_and_logos(image_path, output_path, category, ocr_reader,
                           protect_center=True, extra_margin=2, inpaint_radius=3):
    try:
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            try:
                _pil_read = Image.open(image_path).convert('RGB')
                img_bgr = cv2.cvtColor(np.array(_pil_read), cv2.COLOR_RGB2BGR)
            except Exception as _re:
                return False, 0.0, f"Failed to read image: {_re}"

        original_quality = cv2.Laplacian(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        mask = build_removal_mask(img_bgr, category, ocr_reader, extra_margin, protect_center)
        removed_pct = round(np.sum(mask == 255) / mask.size * 100, 2)
        if removed_pct > 25:
            kernel = np.ones((5, 5), np.uint8)
            mask   = cv2.erode(mask, kernel, iterations=2)
            removed_pct = round(np.sum(mask == 255) / mask.size * 100, 2)
        if np.sum(mask) > 0:
            _img_sf     = max(1.0, max(img_bgr.shape[:2]) / 1000.0)
            safe_radius = min(inpaint_radius, max(5, int(5 * _img_sf)))
            result = cv2.inpaint(img_bgr, mask, inpaintRadius=safe_radius,
                                 flags=cv2.INPAINT_TELEA)
            result_quality = cv2.Laplacian(
                cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if result_quality < original_quality * 0.7:
                result = cv2.inpaint(img_bgr, mask, inpaintRadius=2,
                                     flags=cv2.INPAINT_TELEA)
        else:
            result = img_bgr.copy()

        cv2.imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        return True, removed_pct, None
    except Exception as e:
        return False, 0.0, str(e)

# ===================== WATERMARK =====================
def load_watermark_logo():
    if os.path.exists(WATERMARK_LOGO_FILE):
        try:
            return Image.open(WATERMARK_LOGO_FILE).convert("RGBA")
        except Exception as e:
            st.error(f"Error loading watermark: {e}")
    return None

def save_watermark_logo(uploaded_file):
    try:
        logo = Image.open(uploaded_file).convert("RGBA")
        logo.save(WATERMARK_LOGO_FILE, "PNG")
        return True
    except Exception as e:
        st.error(f"Error saving watermark: {e}")
        return False

def apply_watermark_to_image(image_path, output_path, watermark_logo=None,
                              logo_size_ratio=0.05, logo_opacity=0.20):
    try:
        main_image = Image.open(image_path)
        if main_image.mode not in ('RGB', 'RGBA'):
            main_image = main_image.convert('RGB')
        img_width, img_height = main_image.size
        if watermark_logo is None:
            watermark_logo = load_watermark_logo()
        _fmt = load_app_settings().get("output_format", "WEBP")
        _ext, _pil_fmt, _ = get_format_extension(_fmt)
        _save_kwargs = ({"quality": 100, "subsampling": 0} if _pil_fmt == "JPEG"
                        else {"quality": 100} if _pil_fmt in ("WEBP", "AVIF") else {})
        if watermark_logo is None:
            if main_image.mode == 'RGBA':
                main_image = main_image.convert('RGB')
            main_image.save(output_path, _pil_fmt, **_save_kwargs)
            return True
        logo_width = max(int(img_width * logo_size_ratio), 40)
        logo_aspect_ratio = watermark_logo.height / watermark_logo.width
        logo_height = int(logo_width * logo_aspect_ratio)
        logo_resized = watermark_logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
        if logo_resized.mode == 'RGBA':
            r, g, b, alpha = logo_resized.split()
            alpha = alpha.point(lambda p: int(p * logo_opacity))
            logo_resized = Image.merge('RGBA', (r, g, b, alpha))
        else:
            logo_resized = logo_resized.convert('RGBA')
            alpha = Image.new('L', logo_resized.size, int(255 * logo_opacity))
            logo_resized.putalpha(alpha)
        logo_x = (img_width - logo_width) // 2
        logo_y = (img_height - logo_height) // 2
        watermarked = main_image.convert('RGBA')
        watermarked.paste(logo_resized, (logo_x, logo_y), logo_resized)
        watermarked.convert('RGB').save(output_path, _pil_fmt, **_save_kwargs)
        return True
    except Exception:
        try:
            img = Image.open(image_path).convert('RGB')
            _fmt = load_app_settings().get("output_format", "WEBP")
            _ext, _pil_fmt, _ = get_format_extension(_fmt)
            _save_kwargs = ({"quality": 100, "subsampling": 0} if _pil_fmt == "JPEG"
                            else {"quality": 100} if _pil_fmt in ("WEBP", "AVIF") else {})
            img.save(output_path, _pil_fmt, **_save_kwargs)
        except:
            pass
        return False


# ============================================================================
# AI CLASSIFICATION
# ============================================================================

def predict_image(img_path: str):
    if model is None:
        return 'gallery', 0.5, {'floorplan': 0.33, 'gallery': 0.34, 'masterplan': 0.33}
    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        arr       = np.array(img, dtype=np.float32)
        arr_div   = arr / 255.0
        arr_keras = (arr - 127.5) / 127.5

        def _run(a):
            return model.predict(np.expand_dims(a, 0), verbose=0)[0]

        preds_div   = _run(arr_div)
        preds_keras = _run(arr_keras)
        preds = preds_div if np.max(preds_div) >= np.max(preds_keras) else preds_keras
        idx        = int(np.argmax(preds))
        label      = CLASSES[idx]
        confidence = float(preds[idx])
        all_probs  = {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}
        return label, confidence, all_probs
    except Exception:
        return 'gallery', 0.5, {'floorplan': 0.33, 'gallery': 0.34, 'masterplan': 0.33}





# ===================== RESIZE IMAGE =====================
def resize_image(image_path, target_size, output_path, watermark_logo=None):
    try:
        img = Image.open(image_path).convert('RGB')
        orig_w, orig_h = img.size
        target_w, target_h = target_size
        scale  = min(target_w / orig_w, target_h / orig_h)
        new_w  = int(orig_w * scale)
        new_h  = int(orig_h * scale)
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (target_w, target_h), (255, 255, 255))
        canvas.paste(img_resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
        canvas.save(output_path, "PNG")
        return True
    except Exception:
        return False

# ===================== PROCESS SINGLE IMAGE =====================
def process_single_image(image_path, filename, conf_thresh, qual_thresh, file_size=None,
                          watermark_logo=None, enable_text_removal=True, text_removal_settings=None,
                          enable_sr=False, sr_model_key=None, sr_tile_size=256, sr_tile_overlap=32):
    try:
        label, conf, all_probs = predict_image(image_path)
        quality_score, metrics = assess_image_quality(image_path)

        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        top_conf     = sorted_probs[0][1] if sorted_probs else conf
        top_label    = sorted_probs[0][0] if sorted_probs else label

        if top_conf < conf_thresh:
            try:
                _pil_chk = Image.open(image_path).convert('RGB')
                _chk     = cv2.cvtColor(np.array(_pil_chk), cv2.COLOR_RGB2BGR)
                _gray    = cv2.cvtColor(_chk, cv2.COLOR_BGR2GRAY)
                _var     = cv2.Laplacian(_gray, cv2.CV_64F).var()
                category = top_label if (_var > 30 and top_conf > 0.15) else top_label
            except Exception:
                category = top_label
        else:
            category = top_label

        try:
            with Image.open(image_path) as _img:
                width, height = _img.size
        except Exception:
            width, height = 0, 0

        # ── qual_thresh is now a raw Laplacian variance score (e.g. 500) ─────
        quality_status = "Good Quality" if quality_score >= qual_thresh else "Bad Quality"
        quality_folder = "good_quality" if quality_status == "Good Quality" else "bad_quality"

        output_dir = os.path.join(OUTPUT_DIR, category, quality_folder)
        os.makedirs(output_dir, exist_ok=True)
        base, _    = os.path.splitext(filename)
        _out_fmt   = load_app_settings().get("output_format", "WEBP")
        _out_ext, _, _ = get_format_extension(_out_fmt)

        # Category prefix mapping
        _CATEGORY_PREFIX = {
            "floorplan":  "FLOORPLAN",
            "masterplan": "MASTERPLAN",
            "gallery":    "GALLERY",
            "rejected":   "REJECTED",
        }
        _prefix = _CATEGORY_PREFIX.get(category, "IMAGE")

        # Timestamp-based ID: YYYYMMDD-HHMMSS
        _now       = datetime.now()
        _date_part = _now.strftime('%Y%m%d')  # e.g. 20260314
        _time_part = _now.strftime('%H%M%S')  # e.g. 153045

        # Format: GALLERY-20260314-153045.webp
        new_filename = f"{_prefix}-{_date_part}-{_time_part}.{_out_ext}"
        output_path  = os.path.join(output_dir, new_filename)

        # Safety: if same-second collision, wait 1 second and retry
        if os.path.exists(output_path):
            import time as _t
            _t.sleep(1)
            _now       = datetime.now()
            _date_part = _now.strftime('%Y%m%d')
            _time_part = _now.strftime('%H%M%S')
            new_filename = f"{_prefix}-{_date_part}-{_time_part}.{_out_ext}"
            output_path  = os.path.join(output_dir, new_filename)

        resize_info       = "Original"
        text_removal_info = "Skipped"
        text_removal_pct  = 0.0
        sr_info           = "Skipped"
        temp_files        = []
        current_file      = image_path

        # ── Step 0: Convert any special format to PNG for safe processing ────
        src_ext = os.path.splitext(image_path)[1].lower()
        if src_ext in ('.webp', '.bmp', '.tiff', '.tif', '.gif', '.avif'):
            _conv_path = os.path.join(TEMP_DIR, f"conv_{uuid.uuid4().hex}.png")
            try:
                Image.open(image_path).convert('RGB').save(_conv_path, 'PNG')
                current_file = _conv_path
                temp_files.append(_conv_path)
            except Exception:
                current_file = image_path

        # ── Step 1: TEXT REMOVAL ──────────────────────────────────────────────
        if enable_text_removal and category != "rejected" and ocr_reader is not None:
            text_category = CATEGORY_TO_TEXT_REMOVAL.get(category)
            if text_category:
                temp_cleaned = os.path.join(TEMP_DIR, f"temp_cleaned_{uuid.uuid4().hex}.png")
                tr_settings    = text_removal_settings or {}
                protect_center = tr_settings.get('protect_center', True)
                extra_margin   = tr_settings.get('extra_margin', 2)
                inpaint_radius = tr_settings.get('inpaint_radius', 3)
                success, removed_pct, error = remove_text_and_logos(
                    current_file, temp_cleaned, text_category, ocr_reader,
                    protect_center, extra_margin, inpaint_radius
                )
                if success:
                    text_removal_info = f"Removed {removed_pct}% text/logos"
                    text_removal_pct  = removed_pct
                    temp_files.append(temp_cleaned)
                    current_file = temp_cleaned
                else:
                    text_removal_info = f"Failed: {error}" if error else "Failed"

        # ── Step 2: SUPER-RESOLUTION ──────────────────────────────────────────
        if enable_sr and sr_model_key:
            try:
                torch.set_grad_enabled(False)
                _sr_model, _sr_scale, _sr_dev = load_sr_model(sr_model_key)
                _sr_input = Image.open(current_file).convert("RGB")
                _sr_w, _sr_h = _sr_input.size
                if max(_sr_w, _sr_h) < 2000:
                    _sr_out = run_sr(
                        _sr_model, _sr_scale, _sr_dev,
                        _sr_input, sr_tile_size, sr_tile_overlap
                    )
                    _tmp_sr = os.path.join(TEMP_DIR, f"temp_sr_{uuid.uuid4().hex}.png")
                    _sr_out.save(_tmp_sr, "PNG")
                    temp_files.append(_tmp_sr)
                    current_file = _tmp_sr
                    sr_info = f"SR×{_sr_scale} ({_sr_w}×{_sr_h} → {_sr_w*_sr_scale}×{_sr_h*_sr_scale})"
                else:
                    sr_info = f"Skipped (image already large: {_sr_w}×{_sr_h})"
            except Exception as _e:
                sr_info = f"SR Failed: {str(_e)[:60]}"

        # ── Step 3: RESIZE ────────────────────────────────────────────────────
        if category != "rejected" and CATEGORY_SIZES.get(category):
            temp_resized = os.path.join(TEMP_DIR, f"temp_resized_{uuid.uuid4().hex}.png")
            if resize_image(current_file, CATEGORY_SIZES[category], temp_resized):
                resize_info  = f"{CATEGORY_SIZES[category][0]}×{CATEGORY_SIZES[category][1]}"
                temp_files.append(temp_resized)
                current_file = temp_resized
            else:
                resize_info = "Resize Failed"

        # ── Step 4: WATERMARK + SAVE FINAL ───────────────────────────────────
        apply_watermark_to_image(current_file, output_path, watermark_logo)

        for _tmp in temp_files:
            if _tmp and os.path.exists(_tmp):
                try:
                    os.remove(_tmp)
                except:
                    pass

        return {
            "filename":          filename,
            "file_size":         format_file_size(file_size) if file_size else "N/A",
            "category":          CATEGORY_MAPPING.get(category, category),
            "category_raw":      category,
            "confidence":        round(conf * 100, 2),
            "all_probabilities": {k: round(v * 100, 2) for k, v in all_probs.items()},
            "quality_status":    quality_status,
            "quality_score":     round(quality_score, 2),
            "sharpness":         metrics.get("sharpness",  0),
            "brightness":        metrics.get("brightness", 0),
            "contrast":          metrics.get("contrast",   0),
            "blur":              metrics.get("blur",       0),
            "edge":              metrics.get("edge",       0),
            "width":             width,
            "height":            height,
            "resolution":        f"{width}×{height}",
            "min_resolution":    min(width, height),
            "quality_threshold": qual_thresh,
            "output_size":       resize_info,
            "sr_info":           sr_info,
            "text_removal":      text_removal_info,
            "text_removal_pct":  text_removal_pct,
            "output_path":       output_path,
            "quality_folder":    quality_folder,
            "saved_as":          new_filename,
            "status":            "success"
        }

    except Exception as e:
        return {
            "filename":         filename,
            "file_size":        format_file_size(file_size) if file_size else "N/A",
            "category":         "Error",
            "category_raw":     "error",
            "confidence":       0,
            "quality_status":   "Error",
            "quality_score":    0,
            "resolution":       "N/A",
            "output_size":      "N/A",
            "sr_info":          "Error",
            "text_removal":     "Error",
            "text_removal_pct": 0,
            "status":           "failed",
            "error":            str(e)
        }


# ============================================================================
# STATISTICS
# ============================================================================

def load_statistics():
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        "total_processed": 0, "floorplan_count": 0, "masterplan_count": 0,
        "gallery_count": 0, "rejected_count": 0, "good_quality_count": 0,
        "bad_quality_count": 0, "first_upload_date": None,
        "last_upload_date": None, "total_sessions": 0, "processing_history": []
    }

def save_statistics(stats):
    try:
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
        return True
    except Exception:
        return False

def update_statistics(results):
    stats          = load_statistics()
    new_floorplan  = len([r for r in results if r.get('category_raw') == 'floorplan'  and r.get('status') == 'success'])
    new_masterplan = len([r for r in results if r.get('category_raw') == 'masterplan' and r.get('status') == 'success'])
    new_gallery    = len([r for r in results if r.get('category_raw') == 'gallery'    and r.get('status') == 'success'])
    new_rejected   = len([r for r in results if r.get('category_raw') == 'rejected'   and r.get('status') == 'success'])
    new_good       = len([r for r in results if r.get('quality_status') == 'Good Quality' and r.get('status') == 'success'])
    new_bad        = len([r for r in results if r.get('quality_status') == 'Bad Quality'  and r.get('status') == 'success'])
    total_new      = new_floorplan + new_masterplan + new_gallery + new_rejected
    stats['total_processed']    += total_new
    stats['floorplan_count']    += new_floorplan
    stats['masterplan_count']   += new_masterplan
    stats['gallery_count']      += new_gallery
    stats['rejected_count']     += new_rejected
    stats['good_quality_count'] += new_good
    stats['bad_quality_count']  += new_bad
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if stats['first_upload_date'] is None:
        stats['first_upload_date'] = current_date
    stats['last_upload_date'] = current_date
    stats['total_sessions'] += 1
    stats['processing_history'].append({
        'date': current_date, 'total': total_new,
        'floorplan': new_floorplan, 'masterplan': new_masterplan,
        'gallery': new_gallery, 'rejected': new_rejected,
        'good_quality': new_good, 'bad_quality': new_bad
    })
    if len(stats['processing_history']) > 100:
        stats['processing_history'] = stats['processing_history'][-100:]
    save_statistics(stats)
    return stats



# ============================================================================
# UPLOAD HISTORY
# ============================================================================

def load_upload_history() -> list:
    try:
        if UPLOAD_HISTORY_FILE.exists():
            with open(UPLOAD_HISTORY_FILE, "r") as f:
                data = json.load(f)
            # Handle both dict and list formats
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Old format may store records under a key
                return data.get("records", data.get("history", [])) # type: ignore
    except Exception:
        pass
    return []


def add_upload_record(results: list) -> dict:
    history = load_upload_history()
    record = {
        "timestamp": datetime.now(_IST).isoformat(),
        "total": len(results),
        "successful": len([r for r in results if r.get("status") == "success"]),
        "failed": len([r for r in results if r.get("status") != "success"]),
        "categories": {
            "gallery": len([r for r in results if r.get("category_raw") == "gallery"]),
            "floorplan": len([r for r in results if r.get("category_raw") == "floorplan"]),
            "masterplan": len([r for r in results if r.get("category_raw") == "masterplan"]),
            "rejected": len([r for r in results if r.get("category_raw") == "rejected"]),
        },
    }
    history.append(record)
    try:
        with open(UPLOAD_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass
    return record




########## ZIP DOWNLOAD CREATION ##########


def create_download_zip(results: list, timestamp: str):
    zip_buffer = BytesIO()
    try:
        df           = pd.DataFrame(results)
        total        = len(results)
        successful   = len([r for r in results if r.get('status') == 'success'])
        property_img = len([r for r in results if r.get('category_raw') == 'gallery'])
        floor_img    = len([r for r in results if r.get('category_raw') == 'floorplan'])
        master_img   = len([r for r in results if r.get('category_raw') == 'masterplan'])
        rejected_img = len([r for r in results if r.get('category_raw') == 'rejected'])
        good_qual    = len([r for r in results if r.get('quality_status') == 'Good Quality'])
        bad_qual     = len([r for r in results if r.get('quality_status') == 'Bad Quality'])
        df_success   = df[df['status'] == 'success']
        avg_conf         = df_success['confidence'].mean()       if len(df_success) > 0 else 0
        avg_qual         = df_success['quality_score'].mean()    if len(df_success) > 0 else 0
        avg_text_removal = df_success['text_removal_pct'].mean() if len(df_success) > 0 else 0

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(f"reports/homes247_report_{timestamp}.csv",  df.to_csv(index=False))
            zip_file.writestr(f"reports/homes247_report_{timestamp}.json", json.dumps(results, indent=2))
            images_added = 0
            for result in results:
                if result.get('status') == 'success' and 'output_path' in result:
                    op = result['output_path']
                    if os.path.exists(op):
                        category       = result['category_raw']
                        quality_status = result.get('quality_status', 'Unknown')
                        quality_folder = result.get('quality_folder', 'unknown')
                        # ZIP structure: images/Good Quality/category/filename
                        #            OR images/Bad Quality/category/filename
                        zip_path = f"images/{quality_status}/{category}/{os.path.basename(op)}"
                        with open(op, 'rb') as img_file:
                            zip_file.writestr(zip_path, img_file.read())
                        images_added += 1
            summary_text = f"""HOMES247 - PROCESSING SUMMARY
=========================================
Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Report ID: {timestamp}
PIPELINE ORDER: Text Removal → SR → Resize → Watermark → Save
QUALITY CHECK:  Laplacian Variance (Blur Only, normalised 500×500)
STATISTICS
----------
Total Images: {total}
Successfully Processed: {successful} ({successful/total*100:.1f}%)
Failed: {total-successful}
Images in ZIP: {images_added}
CATEGORIES
----------
Gallery: {property_img}
Floor Plans: {floor_img}
Master Plans: {master_img}
Rejected: {rejected_img}
QUALITY
-------
Good Quality: {good_qual} ({good_qual/total*100:.1f}%)
Bad Quality:  {bad_qual}  ({bad_qual/total*100:.1f}%)
METRICS
-------
Avg Confidence: {avg_conf:.2f}%
Avg Blur Score: {avg_qual:.2f}
Avg Text Removed: {avg_text_removal:.2f}%
OUTPUT SIZES
------------
Floor Plans:  1500×1500
Master Plans: 1640×860
Gallery:      820×430
Rejected:     Original
---
Homes247 - India's Favourite Property Portal
Streamlit Dashboard Version 3.1
"""
            zip_file.writestr("README.txt", summary_text)
        zip_buffer.seek(0)
        return zip_buffer
    except Exception as e:
        st.error(f"ZIP creation failed: {str(e)}")
        return None



# ============================================================================
# SERVER EXPORT
# ============================================================================

def send_results_to_server(results: list, server_url: str, api_key: str = "", extra_headers: dict = None):
    try:
        payload = json.dumps({
            "source": "Homes247-Dashboard-v3.1",
            "timestamp": datetime.now().isoformat(),
            "total": len(results), "results": results
        }).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)
        req = urllib.request.Request(server_url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            status_code = resp.status
            body = resp.read().decode("utf-8", errors="replace")
        if status_code in (200, 201, 202):
            return True, f"✅ Server responded {status_code}: {body[:200]}"
        else:
            return False, f"⚠️ Server responded {status_code}: {body[:200]}"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


# ============================================================================
# DISCOVER SCRAPED DATA (for scraper integration)
# ============================================================================

def discover_scraped_data(output_dir: Path) -> list:
    """Find all current.json files across all scrapers/cities."""
    results = []
    if not output_dir.exists():
        return results
    for scraper_dir in sorted(output_dir.iterdir()):
        if not scraper_dir.is_dir():
            continue
        for city_dir in sorted(scraper_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            current_json = city_dir / "current.json"
            if not current_json.exists():
                continue
            try:
                with open(current_json, "r", encoding="utf-8") as f:
                    props = json.load(f)
                if not isinstance(props, list):
                    continue
                total_images = sum(len(p.get("property_images", []) or []) for p in props)
                total_fps = sum(len(p.get("floor_plans", []) or []) for p in props)
                results.append({
                    "scraper": scraper_dir.name, "city": city_dir.name,
                    "path": str(current_json),
                    "property_count": len(props),
                    "image_count": total_images,
                    "floor_plan_count": total_fps,
                })
            except Exception:
                continue
    return results

def optimize_all_scraped_data(output_dir, optimised_dir, scraper_filter=None,
                              city_filter=None, progress_callback=None, stop_check=None):
    """Placeholder for scraper-based optimization (kept for backward compat)."""
    return {"error": "Use the Image Processing section instead", "sources": []}


# ============================================================================
# CLOUDFLARE R2 AUTO UPLOAD
# ============================================================================

import boto3
from botocore.config import Config

R2_ENDPOINT   = "https://526cb6cff3fef8ee10043692ecb532f8.r2.cloudflarestorage.com"
R2_ACCESS_KEY = "ef33ddac861a32631a5550816f964877"
R2_SECRET_KEY = "86af61f695db174a00398be226efb102d81aea1332d22f72aa7fce6b210b3b0d"
R2_BUCKET     = "img"
R2_SUPPORTED  = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
CONTENT_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png",  ".webp": "image/webp",
    ".gif": "image/gif",  ".bmp": "image/bmp",
    ".tiff": "image/tiff",".tif":  "image/tiff",
}

def get_r2_client():
    try:
        client = boto3.client(
            "s3",
            endpoint_url          = R2_ENDPOINT,
            aws_access_key_id     = R2_ACCESS_KEY,
            aws_secret_access_key = R2_SECRET_KEY,
            config                = Config(signature_version="s3v4",
                                           retries={"max_attempts": 3, "mode": "standard"}),
            region_name = "auto",
        )
        client.list_buckets()   # quick auth test — fails fast if credentials wrong
        return client
    except Exception as e:
        return None

def auto_upload_to_r2(results: list, status_placeholder) -> dict:
    summary = {"uploaded": 0, "failed": 0, "skipped": 0, "errors": []}
    try:
        client = get_r2_client()
    except Exception as e:
        summary["errors"].append(f"R2 client error: {str(e)}")
        return summary
    if client is None:
        summary["errors"].append("R2 connection failed — check credentials and endpoint.")
        return summary

    # ── ONLY upload Good Quality images ──────────────────────────────────
    good_results = [
        r for r in results
        if r.get("status") == "success" and r.get("quality_status") == "Good Quality"
    ]
    total = len(good_results)
    if total == 0:
        summary["errors"].append("No Good Quality images to upload.")
        return summary

    status_placeholder.info(f"☁️ Uploading {total} Good Quality image(s) to R2…")
    r2_progress = st.progress(0)

    for idx, result in enumerate(good_results):
        output_path   = result.get("output_path", "")
        category      = result.get("category_raw", "unknown") or "unknown"
        quality_folder = result.get("quality_folder", "good_quality")
        filename      = os.path.basename(output_path) if output_path else ""

        if not output_path or not filename or not os.path.exists(output_path):
            summary["skipped"] += 1
            r2_progress.progress((idx + 1) / total)
            continue

        status_placeholder.info(f"☁️ Uploading to R2: {idx+1}/{total} — {filename}")

        ext          = os.path.splitext(filename)[1].lower()
        content_type = CONTENT_TYPES.get(ext, "image/jpeg")
        # R2 key structure: category/filename
        r2_key = f"{category}/{filename}"

        try:
            with open(output_path, "rb") as f:
                client.put_object(
                    Bucket      = R2_BUCKET,
                    Key         = r2_key,
                    Body        = f,
                    ContentType = content_type
                )
            summary["uploaded"] += 1
        except Exception as e:
            summary["failed"] += 1
            err_msg = str(e)
            summary["errors"].append(f"{filename}: {err_msg}")
            if "InvalidAccessKeyId" in err_msg or "SignatureDoesNotMatch" in err_msg:
                break

        r2_progress.progress((idx + 1) / total)

    r2_progress.empty()
    return summary
def save_session_report(results: list, processing_time_sec: float, session_start_dt: datetime) -> str:
    try:
        session_id     = session_start_dt.strftime('%Y%m%d_%H%M%S')
        date_str       = session_start_dt.strftime('%Y-%m-%d')
        start_time_str = session_start_dt.strftime('%H:%M:%S')
        end_time_str   = datetime.now().strftime('%H:%M:%S')
        mins           = int(processing_time_sec // 60)
        secs           = int(processing_time_sec % 60)
        total_images   = len(results)
        total_success  = len([r for r in results if r.get('status') == 'success'])
        total_failed   = total_images - total_success
        total_good     = len([r for r in results if r.get('quality_status') == 'Good Quality' and r.get('status') == 'success'])
        total_bad      = len([r for r in results if r.get('quality_status') == 'Bad Quality'  and r.get('status') == 'success'])

        def _build_category(cat_key, cat_label):
            cat_results = [r for r in results if r.get('category_raw') == cat_key]
            processed   = [r for r in cat_results if r.get('status') == 'success']
            rejected    = [r for r in cat_results if r.get('status') != 'success']
            return {
                "category": cat_label,
                "total_images": len(cat_results),
                "total_processed_images": len(processed),
                "rejected_images": len(rejected),
                "images_with_name_and_id": [r.get('saved_as', r.get('filename', 'unknown')) for r in cat_results]
            }

        def _build_others():
            cat_results = [r for r in results
                           if r.get('category_raw') in ('rejected', 'error') or r.get('status') == 'failed']
            return {
                "category": "Others / Rejected",
                "total_images": len(cat_results),
                "total_processed_images": 0,
                "rejected_images": len(cat_results),
                "images_with_name_and_id": [r.get('saved_as', r.get('filename', 'unknown')) for r in cat_results]
            }

        report = {
            "session_id": session_id, "date": date_str,
            "start_time": start_time_str, "end_time": end_time_str,
            "total_processing_time": f"{mins} min {secs} sec",
            "processing_time_seconds": round(processing_time_sec, 2),
            "output_format": load_app_settings().get("output_format", "WEBP"),
            "overall_summary": {
                "total_images": total_images, "total_processed": total_success,
                "total_failed": total_failed, "good_quality": total_good,
                "bad_quality": total_bad,
                "success_rate": f"{round(total_success / total_images * 100, 2)}%" if total_images else "0%"
            },
            "masterplan": _build_category("masterplan", "Master Plan Images"),
            "floorplan":  _build_category("floorplan",  "Floor Plan Images"),
            "gallery":    _build_category("gallery",    "Gallery Images"),
            "others":     _build_others()
        }

        os.makedirs(SESSION_REPORTS_DIR, exist_ok=True)
        session_file = os.path.join(SESSION_REPORTS_DIR, f"session_{session_id}.json")
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        master_log = {"sessions": []}
        if os.path.exists(ALL_SESSIONS_FILE):
            try:
                with open(ALL_SESSIONS_FILE, 'r', encoding='utf-8') as f:
                    master_log = json.load(f)
            except Exception:
                master_log = {"sessions": []}
        master_log["sessions"].append(report)
        with open(ALL_SESSIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(master_log, f, indent=2, ensure_ascii=False)

        return session_file
    except Exception as e:
        return f"ERROR: {str(e)}"