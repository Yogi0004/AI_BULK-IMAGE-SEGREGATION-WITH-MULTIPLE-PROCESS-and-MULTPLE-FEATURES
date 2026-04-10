"""
Homes247 Premium Real Estate Image Processing - Streamlit Dashboard
Version 3.1 - FIXED PIPELINE ORDER: Text Removal → SR → Resize → Watermark → Save
Quality Check: Laplacian Variance (Blur Only) — size-normalised to 500×500
"""

# ── Suppress TensorFlow warnings FIRST ───────────────────────────────────────
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import io
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import shutil
import uuid
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
import time
import zipfile
from io import BytesIO
import tempfile
import urllib.request
import urllib.parse
import threading

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

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
        compress_image_to_150kb(output_path, _fmt)
        return True
    except Exception:
        try:
            img = Image.open(image_path).convert('RGB')
            _fmt = load_app_settings().get("output_format", "WEBP")
            _ext, _pil_fmt, _ = get_format_extension(_fmt)
            _save_kwargs = ({"quality": 100, "subsampling": 0} if _pil_fmt == "JPEG"
                            else {"quality": 100} if _pil_fmt in ("WEBP", "AVIF") else {})
            img.save(output_path, _pil_fmt, **_save_kwargs)
            compress_image_to_150kb(output_path, _fmt)
        except:
            pass
        return False

# ===================== UPLOAD HISTORY =====================
def compress_image_to_150kb(image_path: str, fmt: str) -> bool:
    """
    Compress a saved WEBP or AVIF image to below 150 KB by binary-searching quality.
    Operates in-place. Returns True if compression was applied.
    """
    TARGET_BYTES = 150 * 1024
    if fmt.upper() not in ("WEBP", "AVIF"):
        return False
    try:
        if not os.path.exists(image_path):
            return False
        if os.path.getsize(image_path) <= TARGET_BYTES:
            return False  # already small enough
        img = Image.open(image_path).convert("RGB")
        pil_fmt = "WEBP"   # PIL uses WEBP for both; AVIF needs plugin
        lo, hi, best_quality, best_bytes = 1, 85, 40, None
        while lo <= hi:
            mid = (lo + hi) // 2
            buf = io.BytesIO()
            try:
                img.save(buf, format=pil_fmt, quality=mid, method=6)
            except Exception:
                img.save(buf, format="JPEG", quality=mid, optimize=True)
            size = buf.tell()
            if size <= TARGET_BYTES:
                best_quality = mid
                best_bytes   = buf.getvalue()
                lo = mid + 1   # try higher quality
            else:
                hi = mid - 1   # need lower quality
        # If even quality=1 doesn't fit → scale down dimensions
        if best_bytes is None:
            scale = 0.9
            for _ in range(10):
                nw = max(1, int(img.width  * scale))
                nh = max(1, int(img.height * scale))
                tmp = img.resize((nw, nh), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                tmp.save(buf, format=pil_fmt, quality=40, method=6)
                if buf.tell() <= TARGET_BYTES:
                    best_bytes = buf.getvalue()
                    break
                scale *= 0.85
        if best_bytes:
            with open(image_path, "wb") as f:
                f.write(best_bytes)
            return True
        return False
    except Exception:
        return False

def load_upload_history():
    if os.path.exists(UPLOAD_HISTORY_FILE):
        try:
            with open(UPLOAD_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"uploads": []}

def save_upload_history(history):
    try:
        with open(UPLOAD_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
        return True
    except Exception:
        return False

def add_upload_record(results):
    history        = load_upload_history()
    new_floorplan  = len([r for r in results if r.get('category_raw') == 'floorplan'  and r.get('status') == 'success'])
    new_masterplan = len([r for r in results if r.get('category_raw') == 'masterplan' and r.get('status') == 'success'])
    new_gallery    = len([r for r in results if r.get('category_raw') == 'gallery'    and r.get('status') == 'success'])
    new_rejected   = len([r for r in results if r.get('category_raw') == 'rejected'   and r.get('status') == 'success'])
    new_good       = len([r for r in results if r.get('quality_status') == 'Good Quality' and r.get('status') == 'success'])
    new_bad        = len([r for r in results if r.get('quality_status') == 'Bad Quality'  and r.get('status') == 'success'])
    successful     = len([r for r in results if r.get('status') == 'success'])
    upload_record  = {
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "total": successful, "floorplan": new_floorplan, "masterplan": new_masterplan,
        "gallery": new_gallery, "rejected": new_rejected,
        "good_quality": new_good, "bad_quality": new_bad
    }
    history["uploads"].append(upload_record)
    save_upload_history(history)
    return upload_record

# ===================== STATISTICS =====================
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

# ===================== SESSION REPORT =====================
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

# ===================== CSS =====================
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #1e0f32 0%, #0f0520 100%); }
    h1, h2, h3 { color: #e0e7ff !important; font-weight: 700; }
    .stAlert, .stExpander {
        background: rgba(30, 15, 50, 0.6);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 10px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 2rem; font-weight: 600; transition: all 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);
        box-shadow: 0 8px 20px rgba(139, 92, 246, 0.4);
        transform: translateY(-2px);
    }
    [data-testid="stMetricValue"] { color: #8b5cf6; font-size: 2rem; font-weight: 800; }
    .stProgress > div > div { background: linear-gradient(90deg, #8b5cf6 0%, #7c3aed 100%); }
    .img-panel{background:rgba(30,15,50,0.6);border:1px solid rgba(139,92,246,0.3);
      border-radius:12px;overflow:hidden;margin-bottom:1rem;}
    .img-panel-header{padding:0.7rem 1rem;border-bottom:1px solid rgba(139,92,246,0.3);}
    .img-panel-badge{font-size:0.62rem;letter-spacing:0.18em;text-transform:uppercase;
      padding:0.2rem 0.6rem;border-radius:20px;font-weight:600;}
    .badge-input{background:rgba(122,125,138,0.18);color:#7a7d8a;}
    .badge-output{background:rgba(139,92,246,0.18);color:#8b5cf6;}
    .img-panel-body{padding:1rem;}
    .stats-row{display:flex;gap:0.8rem;flex-wrap:wrap;margin-top:0.6rem;}
    .stat-chip{background:rgba(255,255,255,0.04);border:1px solid rgba(139,92,246,0.3);
      border-radius:8px;padding:0.4rem 0.8rem;font-size:0.78rem;color:#c4b5fd;display:inline-block;}
    .stat-chip b{color:#e0e7ff;font-weight:500;}
    .status-bar{background:rgba(30,15,50,0.6);border:1px solid rgba(139,92,246,0.3);
      border-left:3px solid #8b5cf6;border-radius:12px;padding:1rem 1.2rem;
      font-size:0.88rem;color:#c4b5fd;margin:1rem 0;}
    .status-bar b{color:#8b5cf6;}
</style>
""", unsafe_allow_html=True)

# ===================== PREDICT IMAGE =====================
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

# ===================== ZIP =====================
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

# ===================== SEND TO SERVER =====================
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


# ===================== CLOUDFLARE R2 AUTO-UPLOAD =====================
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

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("""
<div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); border-radius: 10px; margin-bottom: 1rem;'>
    <h2 style='margin: 0; color: white;'>⚙️ Configuration</h2>
</div>
""", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
<div style='background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
    <h3 style='margin: 0 0 0.5rem 0; color: #8b5cf6;'>🏠 Watermark Logo</h3>
    <p style='margin: 0; font-size: 0.9rem; color: #c4b5fd;'>Upload your logo for automatic watermarking</p>
</div>
""", unsafe_allow_html=True)

    watermark_upload = st.file_uploader(
        "Upload Logo (PNG recommended)", type=['png', 'jpg', 'jpeg'],
        key="watermark_uploader", help="Logo applied to center at 20% opacity"
    )
    if watermark_upload:
        if save_watermark_logo(watermark_upload):
            st.success("✅ Watermark logo saved!")
            st.rerun()

    current_logo = load_watermark_logo()
    if current_logo:
        st.info("✓ Watermark: ACTIVE")
        st.image(current_logo, width=100, caption="Current Logo")
        if st.button("🗑️ Remove Watermark", use_container_width=True):
            if os.path.exists(WATERMARK_LOGO_FILE):
                os.remove(WATERMARK_LOGO_FILE)
                st.success("Watermark removed!")
                st.rerun()
    else:
        st.warning("⚠️ No watermark logo uploaded")

    st.markdown("---")

    with st.expander("🎬 Output Resolution Quality", expanded=False):
        if 'selected_resolution' not in st.session_state:
            st.session_state.selected_resolution = "480p"
        resolution_options = {
            "144p":  {"label": "144p",  "desc": "Very Low",  "w": 256,  "h": 144,  "icon": "🔴"},
            "240p":  {"label": "240p",  "desc": "Low",       "w": 426,  "h": 240,  "icon": "🟠"},
            "360p":  {"label": "360p",  "desc": "SD",        "w": 640,  "h": 360,  "icon": "🟡"},
            "480p":  {"label": "480p",  "desc": "SD+",       "w": 854,  "h": 480,  "icon": "🟢"},
            "720p":  {"label": "720p",  "desc": "HD",        "w": 1280, "h": 720,  "icon": "🔵"},
            "1080p": {"label": "1080p", "desc": "Full HD",   "w": 1920, "h": 1080, "icon": "💜"},
            "1440p": {"label": "1440p", "desc": "2K / QHD",  "w": 2560, "h": 1440, "icon": "💎"},
            "2160p": {"label": "2160p", "desc": "4K / UHD",  "w": 3840, "h": 2160, "icon": "👑"},
        }
        for res_key, res_val in resolution_options.items():
            if st.button(
                f"{res_val['icon']} {res_val['label']} — {res_val['desc']} ({res_val['w']}×{res_val['h']})",
                key=f"res_btn_{res_key}", use_container_width=True
            ):
                st.session_state.selected_resolution = res_key
                st.rerun()
        active = resolution_options[st.session_state.selected_resolution]
        st.success(f"✅ Active: {active['icon']} {active['label']} — {active['w']}×{active['h']}px")

    st.markdown("---")

    _current_settings = load_app_settings()
    _format_options   = ["WEBP", "JPEG", "JPG", "PNG", "AVIF"]
    _saved_format     = _current_settings.get("output_format", "WEBP")
    _saved_index      = _format_options.index(_saved_format) if _saved_format in _format_options else 0
    selected_format   = st.selectbox("🖼️ Output File Format", options=_format_options,
                                     index=_saved_index, key="output_format_selector")
    if selected_format != _saved_format:
        _s = load_app_settings()
        _s["output_format"] = selected_format
        save_app_settings(_s)
        st.success(f"✅ Format saved: {selected_format}")

    st.markdown("---")

    enable_text_removal = st.checkbox("🤖 Enable Text & Logo Removal", value=True)
    if enable_text_removal:
        if ocr_reader is not None:
            st.success("✅ OCR Engine Ready")
        else:
            st.error("❌ OCR not available")
        protect_center = st.checkbox("🛡️ Protect Central Content", value=True)
        extra_margin   = st.slider("🔍 Detection Aggressiveness", 0, 5, 2)
        inpaint_radius = st.slider("🎨 Inpaint Smoothness", 1, 8, 3)
    else:
        protect_center = True
        extra_margin   = 2
        inpaint_radius = 3

    st.markdown("---")

    _saved_settings = load_app_settings()
    _saved_conf     = _saved_settings.get("confidence_threshold", 45)

    confidence_threshold = st.slider("🎯 Confidence Threshold", 0, 100, _saved_conf, 5)
    if confidence_threshold != _saved_conf:
        _s = load_app_settings()
        _s["confidence_threshold"] = confidence_threshold
        save_app_settings(_s)
        st.success(f"✅ Confidence saved: {confidence_threshold}%")

    st.markdown("---")

    # ── REPLACED: old quality selectbox → blur threshold slider ───────────────
    st.markdown("""
<div style='background: rgba(139, 92, 246, 0.1); padding: 0.8rem 1rem;
     border-radius: 8px; margin-bottom: 0.8rem;'>
    <h3 style='margin: 0 0 0.3rem 0; color: #8b5cf6;'>🔍 Image Quality (Blur Check)</h3>
    <p style='margin: 0; font-size: 0.82rem; color: #c4b5fd;'>
        Laplacian Variance — score ≥ threshold = Good Quality
    </p>
</div>
""", unsafe_allow_html=True)

    _saved_blur_thresh = load_app_settings().get("blur_threshold", 500)
    quality_threshold  = st.slider(
        "Blur Threshold", min_value=100, max_value=2000,
        value=_saved_blur_thresh, step=50,
        help=(
            "Score ≥ threshold → Good Quality ✅  |  Score < threshold → Bad Quality ❌\n\n"
            "Calibrated values:\n"
            "• Sharp floor plan  → ~2800\n"
            "• Blurred overlay   → ~200–300\n"
            "• Recommended       → 500"
        )
    )
    if quality_threshold != _saved_blur_thresh:
        _s = load_app_settings()
        _s["blur_threshold"] = quality_threshold
        save_app_settings(_s)
        st.success(f"✅ Blur threshold saved: {quality_threshold}")

    st.markdown(f"""
<div style='background:rgba(30,15,50,0.6);border:1px solid rgba(139,92,246,0.3);
     border-radius:8px;padding:0.6rem 0.9rem;font-size:0.75rem;
     font-family:monospace;line-height:1.9;margin-top:0.3rem;'>
    <span style='color:#22c55e;'>✅ GOOD</span>&nbsp; score ≥ {quality_threshold}<br>
    <span style='color:#ef4444;'>❌ BAD &nbsp;</span> score &lt; {quality_threshold}
</div>
""", unsafe_allow_html=True)
    # ── END blur threshold slider ─────────────────────────────────────────────

    st.markdown("---")

    st.markdown("""
<div style='background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
    <h3 style='margin: 0 0 0.5rem 0; color: #8b5cf6;'>🔬 Super-Resolution (Batch)</h3>
    <p style='margin: 0; font-size: 0.9rem; color: #c4b5fd;'>Real-ESRGAN — runs AFTER text removal</p>
</div>
""", unsafe_allow_html=True)

    _sr_model_options  = list(SR_MODELS.keys())
    _sr_from_file      = load_app_settings().get(
                             "sr_model_key", "🏠 Real-ESRGAN ×4  (Best for Photos)")
    if _sr_from_file not in _sr_model_options:
        _sr_from_file  = _sr_model_options[0]
    _sr_saved_index    = _sr_model_options.index(_sr_from_file)
    sr_model_key       = st.selectbox("SR Model", _sr_model_options,
                                      index=_sr_saved_index)
    if sr_model_key != _sr_from_file:
        _s = load_app_settings()
        _s["sr_model_key"] = sr_model_key
        save_app_settings(_s)
        st.success(f"✅ SR Model saved permanently: {sr_model_key}")

    _sr_batch_saved = load_app_settings().get("enable_batch_sr", False)
    enable_batch_sr = st.checkbox("⚡ Apply SR to batch images",
                                  value=_sr_batch_saved,
                                  help="Slow on CPU. Best with GPU.")
    if enable_batch_sr != _sr_batch_saved:
        _s = load_app_settings()
        _s["enable_batch_sr"] = enable_batch_sr
        save_app_settings(_s)

    with st.expander("🔧 SR Inference Settings", expanded=False):
        tile_size    = st.slider("Tile size (pixels)", 64, 512, 256, 64,
                                 help="256 recommended for CPU.", key="sr_tile_size")
        tile_overlap = st.slider("Tile overlap", 16, 128, 32, 16,
                                 help="32 gives smooth seams.", key="sr_tile_overlap")
    _sr_dev     = get_sr_device()
    _scale_info = SR_MODELS[sr_model_key]["scale"]
    _d_icon     = "⚡" if str(_sr_dev) != "cpu" else "💻"
    st.markdown(f"**Device:** {_d_icon} `{str(_sr_dev).upper()}`")
    st.markdown(f"**Upscale:** `{_scale_info}×`")
    if str(_sr_dev) == "cpu":
        st.warning("⚠️ SR on CPU: 3–10 min per image.")
    st.markdown("""
    **Pipeline Order:**
    1. 🧹 Text Removal (original size)
    2. 🔬 Super-Resolution (upscale clean image)
    3. 📐 Resize to target
    4. 💧 Watermark + Save

    **Model Guide:**
    - 🏠 **×4** — Property photos, interiors, exteriors
    - 🏢 **×2** — Gentle upscale, preserves original
    - 🖼️ **×4 Anime** — CGI renders, floorplans
    """)

    st.markdown("---")

    if os.path.exists(ALL_SESSIONS_FILE):
        try:
            with open(ALL_SESSIONS_FILE, 'r') as f:
                all_sess = json.load(f)
            sessions_list = all_sess.get("sessions", [])
            st.success(f"✅ {len(sessions_list)} session(s) logged")
            if sessions_list:
                latest = sessions_list[-1]
                st.caption(f"Latest: {latest.get('date','N/A')} {latest.get('start_time','')}")
        except Exception:
            st.info("No sessions yet.")
    else:
        st.info("No sessions yet.")

    st.markdown("---")
    if st.button("🔄 Reset All Statistics", use_container_width=True):
        for _f in [STATS_FILE, UPLOAD_HISTORY_FILE]:
            if os.path.exists(_f):
                os.remove(_f)
        st.success("Statistics reset!")
        st.rerun()

# ===================== MAIN HEADER =====================
st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); border-radius: 15px; margin-bottom: 2rem;'>
    <h1 style='margin: 0; color: white; font-size: 3rem;'>🏠 HOMES247</h1>
    <p style='margin: 0.5rem 0 0 0; color: #e9d5ff; font-size: 1.2rem;'>India's Favourite Property Portal - AI Dashboard</p>
    <p style='margin: 0.5rem 0 0 0; color: #fae8ff; font-size: 0.9rem;'>∞ UNLIMITED ✓ QUALITY-CHECK ✓ TEXT REMOVAL→SR→RESIZE ✓ AI TEXT REMOVAL ✓ AUTO WATERMARK ✓ SERVER EXPORT ✓ AUTO SESSION REPORTS ✓ SUPER-RESOLUTION ✨</p>
</div>
""", unsafe_allow_html=True)

# ===================== STATISTICS BANNER =====================
stats = load_statistics()
if stats['total_processed'] > 0:
    st.markdown(f"""
<div style='background: rgba(30, 15, 50, 0.6); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(139, 92, 246, 0.3); margin-bottom: 2rem;'>
    <h2 style='text-align: center; color: #8b5cf6; margin: 0 0 1rem 0;'>📊 ALL-TIME PROCESSING STATISTICS</h2>
    <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;'>
        <div style='text-align: center; background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 8px;'>
            <div style='font-size: 2rem; color: #8b5cf6; font-weight: bold;'>{stats['floorplan_count']:,}</div>
            <div style='color: #c4b5fd;'>🏠 Floor Plans</div>
        </div>
        <div style='text-align: center; background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 8px;'>
            <div style='font-size: 2rem; color: #8b5cf6; font-weight: bold;'>{stats['masterplan_count']:,}</div>
            <div style='color: #c4b5fd;'>🗺️ Master Plans</div>
        </div>
        <div style='text-align: center; background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 8px;'>
            <div style='font-size: 2rem; color: #8b5cf6; font-weight: bold;'>{stats['gallery_count']:,}</div>
            <div style='color: #c4b5fd;'>🖼️ Gallery</div>
        </div>
        <div style='text-align: center; background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 8px;'>
            <div style='font-size: 2rem; color: #8b5cf6; font-weight: bold;'>{stats['total_processed']:,}</div>
            <div style='color: #c4b5fd;'>📊 Total Processed</div>
        </div>
    </div>
    <div style='text-align: center; margin-top: 1rem; color: #c4b5fd; font-size: 0.9rem;'>
        📅 First: {stats['first_upload_date'] or 'N/A'} | 🕐 Last: {stats['last_upload_date'] or 'N/A'} | 🔢 Sessions: {stats['total_sessions']}
    </div>
</div>
""", unsafe_allow_html=True)

# ===================== SESSION STATE =====================
if 'results'              not in st.session_state: st.session_state.results = []
if 'processed'            not in st.session_state: st.session_state.processed = False
if 'processing'           not in st.session_state: st.session_state.processing = False

# ── Resume support: restore processing state from query params on refresh ─────
if 'processing_restored' not in st.session_state:
    st.session_state.processing_restored = False
    _qp = st.query_params
    if _qp.get("processing") == "1" and not st.session_state.processing:
        _resume_city    = _qp.get("city", "")
        _resume_scraper = _qp.get("scraper", "")
        _resume_done    = int(_qp.get("done", "0"))
        _resume_total   = int(_qp.get("total", "0"))
        if _resume_total > 0 and _resume_done < _resume_total:
            st.session_state.processing_restored = True
            st.session_state["resume_city"]       = _resume_city
            st.session_state["resume_scraper"]    = _resume_scraper
            st.session_state["resume_done"]       = _resume_done
            st.session_state["resume_total"]      = _resume_total
if 'url_downloaded_paths' not in st.session_state: st.session_state.url_downloaded_paths = []
if 'scan_results'         not in st.session_state: st.session_state.scan_results = []
if 'scanned_images'       not in st.session_state: st.session_state.scanned_images = []
if 'api_fetched_urls'     not in st.session_state: st.session_state.api_fetched_urls = []
if 'last_session_file'    not in st.session_state: st.session_state.last_session_file = None
if 'session_start_dt'     not in st.session_state: st.session_state.session_start_dt = None
if 'sr_img'               not in st.session_state: st.session_state.sr_img = None
if 'sr_bytes'             not in st.session_state: st.session_state.sr_bytes = None
if 'sr_scale'             not in st.session_state: st.session_state.sr_scale = None
if 'sr_model_key_used'    not in st.session_state: st.session_state.sr_model_key_used = None
if 'sr_stem'              not in st.session_state: st.session_state.sr_stem = None

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
ALLOWED_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/webp",
    "image/bmp", "image/tiff", "image/gif", "image/x-tiff",
    "image/x-bmp", "image/x-ms-bmp"
}

# ── Show resume banner if processing was interrupted ─────────────────────────
if st.session_state.get("processing_restored"):
    _r_done  = st.session_state.get("resume_done", 0)
    _r_total = st.session_state.get("resume_total", 0)
    _r_city  = st.session_state.get("resume_city", "")
    st.warning(
        f"⚡ **Processing was interrupted!** "
        f"{_r_done}/{_r_total} images done for **{_r_city.replace('_',' ').title()}**. "
        f"Select the same city and click Process to resume from image {_r_done + 1}."
    )

# ===================== UPLOAD SECTION =====================

input_tab1, input_tab2, input_tab3 = st.tabs([
    "📤 Upload Files",
    "🔗 Download from URLs",
    "🔬 Super-Resolution (Single Image)"
])

# ── TAB 1: Upload Files ────────────────────────────────────────────────────────
with input_tab1:
    uploaded_files = st.file_uploader(
        "Drag & Drop or Browse — JPG, PNG, WEBP, BMP, TIFF, GIF supported",
        type=None, accept_multiple_files=True
    )
    if uploaded_files:
        filtered_files = []
        rejected_files = []
        for uf in uploaded_files:
            ext  = os.path.splitext(uf.name)[1].lower()
            mime = uf.type.lower() if uf.type else ""
            if ext in ALLOWED_EXTENSIONS or mime in ALLOWED_TYPES:
                filtered_files.append(uf)
            else:
                rejected_files.append(uf.name)
        if rejected_files:
            st.warning(f"⚠️ Skipped {len(rejected_files)} unsupported file(s): {', '.join(rejected_files)}")
        uploaded_files = filtered_files

    if uploaded_files:
        total_images = len(uploaded_files)
        total_size   = sum(f.size for f in uploaded_files)
        st.success(f"✅ **{total_images} images** ({format_file_size(total_size)}) ready!")
        estimated_time = total_images * (8 if enable_batch_sr else 2)
        st.info(f"⏱️ Estimated time: ~{estimated_time // 60} min {estimated_time % 60} sec"
                + (" (SR enabled)" if enable_batch_sr else ""))

        if st.button("🚀 PROCESS ALL IMAGES", use_container_width=True,
                     disabled=st.session_state.processing):
            st.session_state.results           = []
            st.session_state.processed         = False
            st.session_state.processing        = True
            st.session_state.last_session_file = None
            session_start                      = datetime.now()
            st.session_state.session_start_dt  = session_start

            watermark_logo        = load_watermark_logo()
            text_removal_settings = {'protect_center': protect_center,
                                     'extra_margin': extra_margin,
                                     'inpaint_radius': inpaint_radius}
            progress_bar  = st.progress(0)
            status_text   = st.empty()
            start_time    = time.time()
            success_count = 0
            failed_count  = 0

            for idx, file in enumerate(uploaded_files):
                current_num = idx + 1
                file_size   = file.size
                status_text.info(f"⏳ Processing {current_num}/{total_images}: {file.name}")
                try:
                    temp_path = os.path.join(UPLOAD_DIR, file.name)
                    with open(temp_path, "wb") as fh:
                        fh.write(file.getbuffer())
                    result = process_single_image(
                        temp_path, file.name,
                        confidence_threshold / 100, quality_threshold,
                        file_size, watermark_logo,
                        enable_text_removal, text_removal_settings,
                        enable_sr=enable_batch_sr,
                        sr_model_key=sr_model_key if enable_batch_sr else None,
                        sr_tile_size=tile_size,
                        sr_tile_overlap=tile_overlap
                    )
                    st.session_state.results.append(result)
                    if result['status'] == 'success':
                        success_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    failed_count += 1
                    st.session_state.results.append({
                        "filename": file.name, "file_size": format_file_size(file_size),
                        "category": "Error", "category_raw": "error",
                        "confidence": 0, "quality_status": "Error", "quality_score": 0,
                        "resolution": "N/A", "output_size": "N/A", "sr_info": "Error",
                        "text_removal": "Error", "text_removal_pct": 0,
                        "status": "failed", "error": str(e)
                    })
                progress_bar.progress(current_num / total_images)

            processing_time = time.time() - start_time
            update_statistics(st.session_state.results)
            add_upload_record(st.session_state.results)
            session_file = save_session_report(st.session_state.results, processing_time, session_start)
            st.session_state.last_session_file = session_file
            st.session_state.processed  = True
            st.session_state.processing = False
            status_text.empty()
            progress_bar.empty()
            st.balloons()
            st.success(f"🎉 Done! ✅ {success_count} success | ❌ {failed_count} failed | ⏱️ {processing_time//60:.0f}m {processing_time%60:.0f}s")
            _r2_ph  = st.empty()
            _r2_res = auto_upload_to_r2(st.session_state.results, _r2_ph)
            _r2_ph.empty()

            # Count good quality images
            good_images_count = len([
                r for r in st.session_state.results
                if r.get("status") == "success" and r.get("quality_status") == "Good Quality"
            ])
            st.info(f"☁️ Good Quality Images Sent to Cloudflare: **{good_images_count}**")
            if _r2_res["failed"] == 0 and _r2_res["uploaded"] > 0:
                st.success(f"☁️ R2 Upload Complete — {_r2_res['uploaded']} file(s) sent to R2.")
            elif _r2_res["uploaded"] > 0:
                st.warning(f"☁️ R2 Partial — ✅ {_r2_res['uploaded']} uploaded | ❌ {_r2_res['failed']} failed")
            else:
                st.error(f"☁️ R2 Upload Failed — {'; '.join(_r2_res['errors'][:3])}")

            time.sleep(2)
            st.rerun()

# ── TAB 2: Download from URLs ──────────────────────────────────────────────────
with input_tab2:
    custom_save_dir = st.text_input("📂 Custom Save Directory (optional)",
                                    placeholder="Leave empty to use default")
    input_type = st.radio("Input Type",
                          options=["📁 Local Folder Path", "🔗 Direct URLs (one per line)", "🌐 API Endpoint", "📦 Scraper JSON Files"],
                          horizontal=True)
    folder_path_tab2  = ""
    url_list_tab2     = []
    scraper_json_urls = []

    if input_type == "📁 Local Folder Path":
        folder_path_tab2 = st.text_input("Enter folder path",
                                          placeholder="e.g., C:/Users/YourName/Pictures")
        if folder_path_tab2 and os.path.isdir(folder_path_tab2):
            imgs = [f for f in os.listdir(folder_path_tab2)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.gif'))]
            st.success(f"Found **{len(imgs)}** images") if imgs else st.warning("No images found.")

    elif input_type == "🔗 Direct URLs (one per line)":
        url_text_t2   = st.text_area("Paste image URLs (one per line)", height=180)
        url_list_tab2 = [l.strip() for l in url_text_t2.splitlines() if l.strip()]
        if url_list_tab2:
            st.caption(f"Found **{len(url_list_tab2)}** URLs")

    elif input_type == "🌐 API Endpoint":
        api_url_t2 = st.text_input("Enter API endpoint URL")
        if api_url_t2 and st.button("🔍 Fetch from API", key="fetch_api_t2"):
            try:
                with urllib.request.urlopen(urllib.request.Request(api_url_t2), timeout=10) as resp:
                    data_api = json.loads(resp.read().decode('utf-8'))
                image_urls_api = data_api.get('images', []) or data_api.get('urls', []) or data_api
                if isinstance(image_urls_api, list) and image_urls_api:
                    st.session_state.api_fetched_urls = [str(u) for u in image_urls_api if isinstance(u, str)]
                    st.success(f"Fetched **{len(st.session_state.api_fetched_urls)}** URLs")
                else:
                    st.error("No valid image URL list found.")
            except Exception as e_api:
                st.error(f"Failed: {str(e_api)}")

    elif input_type == "📦 Scraper JSON Files":
        # Auto-discover scraper output structure: output/{scraper}/{city}/current.json
        _scraper_output_root = Path("output")
        _scrapers_found = []
        if _scraper_output_root.exists():
            for _sd in sorted(_scraper_output_root.iterdir()):
                if _sd.is_dir():
                    _scrapers_found.append(_sd.name)

        if not _scrapers_found:
            st.warning("⚠️ No scraper output found. Run a scraper first from the **🏗️ Scraper** tab.")
        else:
            _sel_scraper_src = st.selectbox(
                "🕸️ Scraper Source",
                ["All Scrapers"] + _scrapers_found,
                key="img_opt_scraper_src",
                format_func=lambda x: x.title() if x != "All Scrapers" else x
            )

            # Discover cities under selected scraper(s)
            _city_map = {}  # city_name → {count, image_count, scrapers}
            for _sd in _scraper_output_root.iterdir():
                if not _sd.is_dir():
                    continue
                if _sel_scraper_src != "All Scrapers" and _sd.name != _sel_scraper_src:
                    continue
                for _cd in sorted(_sd.iterdir()):
                    if not _cd.is_dir():
                        continue
                    _cj = _cd / "current.json"
                    if not _cj.exists():
                        continue
                    try:
                        with open(_cj, "r", encoding="utf-8") as _cf:
                            _cprops = json.load(_cf)
                        if not isinstance(_cprops, list) or not _cprops:
                            continue
                        _cn = _cd.name
                        if _cn not in _city_map:
                            _city_map[_cn] = {
                                "count": 0,
                                "image_count": 0,
                                "optimised_count": 0,
                                "analyzed_count": 0,
                                "analyzed_image_count": 0,
                                "bad_quality_image_count": 0,
                                "total_image_count": 0,
                                "scrapers": []
                            }
                        _city_map[_cn]["count"] += len(_cprops)
                        _city_map[_cn]["scrapers"].append(_sd.name)
                        for _p in _cprops:
                            _p_imgs = len(_p.get("property_images", []))
                            _opt_status = (
                                _p.get("optimised")
                                or _p.get("optimized")
                                or "no"
                            )
                            # Normalize: treat "done"/"analyzed"/"no" only
                            if _opt_status not in ("done", "analyzed"):
                                _opt_status = "no"

                            # Total images — ALL properties
                            _city_map[_cn]["total_image_count"] = _city_map[_cn].get("total_image_count", 0) + _p_imgs

                            if _opt_status == "done":
                                _city_map[_cn]["optimised_count"]       = _city_map[_cn].get("optimised_count", 0) + 1
                                _city_map[_cn]["optimised_image_count"] = _city_map[_cn].get("optimised_image_count", 0) + _p_imgs
                                continue

                            if _opt_status == "analyzed":
                                _city_map[_cn]["analyzed_count"]          = _city_map[_cn].get("analyzed_count", 0) + 1
                                _city_map[_cn]["analyzed_image_count"]    = _city_map[_cn].get("analyzed_image_count", 0) + _p_imgs
                                _city_map[_cn]["bad_quality_image_count"] = _city_map[_cn].get("bad_quality_image_count", 0) + _p_imgs
                                _city_map[_cn]["total_image_count"]       = _city_map[_cn].get("total_image_count", 0) + _p_imgs
                                continue

                            # Pending properties only
                            _city_map[_cn]["image_count"] += _p_imgs

                            for _fp in _p.get("floor_plans", []):
                                if isinstance(_fp, str):
                                    _city_map[_cn]["image_count"] += 1
                                elif isinstance(_fp, dict) and (_fp.get("image_url") or _fp.get("image_url_2d")):
                                    _city_map[_cn]["image_count"] += 1
                    except Exception:
                        continue

            if not _city_map:
                st.info("📭 No scraped data found for the selected source.")
            else:
                _sel_city_src = st.selectbox(
                    "🏙️ City",
                    sorted(_city_map.keys()),
                    key="img_opt_city_src",
                    format_func=lambda x: (
                        f"{x.replace('_',' ').title()} — "
                        f"{_city_map[x]['count']} props, "
                        f"{_city_map[x]['image_count']} pending imgs "
                        f"({', '.join(_city_map[x]['scrapers'])})"
                    )
                )
                _cd_info = _city_map[_sel_city_src]
                # ── Read ALL counts DIRECTLY from current.json every time ─
                _total_props      = 0
                _total_imgs       = 0
                _optimised_cnt    = 0
                _optimised_imgs   = 0   # filenames only = uploaded to R2
                _analyzed_cnt     = 0
                _analyzed_imgs    = 0   # images from "analyzed" props only
                _bad_quality_imgs = 0   # same = not uploaded to R2
                _pending_cnt      = 0
                _pending_imgs     = 0

                for _sd_r in _scraper_output_root.iterdir():
                    if not _sd_r.is_dir():
                        continue
                    if _sel_scraper_src != "All Scrapers" and _sd_r.name != _sel_scraper_src:
                        continue
                    for _cd_r in _sd_r.iterdir():
                        if not _cd_r.is_dir():
                            continue
                        if _cd_r.name != _sel_city_src:
                            continue
                        _cj_r = _cd_r / "current.json"
                        if not _cj_r.exists():
                            continue
                        try:
                            with open(_cj_r, "r", encoding="utf-8") as _fr:
                                _props_r = json.load(_fr)
                            for _pr in _props_r:
                                _opt_raw   = _pr.get("optimised") or _pr.get("optimized") or "no"
                                _opt       = str(_opt_raw).strip().lower()
                                # Normalize values
                                if _opt not in ("done", "analyzed"):
                                    _opt = "no"
                                _prop_imgs = _pr.get("property_images", [])
                                _pimgs     = len(_prop_imgs)
                                _total_props += 1
                                _total_imgs  += _pimgs

                                if _opt == "done":
                                    _optimised_cnt  += 1
                                    # Only filenames = actually in Cloudflare R2
                                    for _img in _prop_imgs:
                                        if isinstance(_img, str) and not _img.startswith("http"):
                                            _optimised_imgs += 1

                                elif _opt == "analyzed":
                                    _analyzed_cnt     += 1
                                    _analyzed_imgs    += _pimgs
                                    _bad_quality_imgs += _pimgs

                                else:
                                    # "no" = pending
                                    _pending_cnt += 1
                                    _pending_imgs += _pimgs

                        except Exception:
                            pass
                st.markdown(
                    f"<div style='background:rgba(139,92,246,0.1);padding:1rem;border-radius:8px;margin:.4rem 0;'>"
                    f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:0.5rem;text-align:center;'>"

                    f"<div style='background:rgba(139,92,246,0.15);border-radius:8px;padding:0.7rem;'>"
                    f"<div style='color:#8b5cf6;font-size:1.5rem;font-weight:700;'>{_total_props}</div>"
                    f"<div style='color:#8b5cf6;font-size:0.8rem;font-weight:600;'>Properties</div>"
                    f"<div style='color:#8b5cf6;font-size:0.78rem;'>({_total_imgs} imgs)</div>"
                    f"<div style='color:#c4b5fd;font-size:0.75rem;margin-top:0.3rem;'>🏠 Total Properties</div>"
                    f"</div>"

                    f"<div style='background:rgba(16,185,129,0.15);border-radius:8px;padding:0.7rem;'>"
                    f"<div style='color:#10b981;font-size:1.5rem;font-weight:700;'>{_optimised_cnt}</div>"
                    f"<div style='color:#10b981;font-size:0.8rem;font-weight:600;'>Properties</div>"
                    f"<div style='color:#10b981;font-size:0.78rem;'>({_optimised_imgs} Imgs ☁️ in R2)</div>"
                    f"<div style='color:#c4b5fd;font-size:0.75rem;margin-top:0.3rem;'>✅ Done Properties</div>"
                    f"</div>"

                    f"<div style='background:rgba(167,139,250,0.15);border-radius:8px;padding:0.7rem;'>"
                    f"<div style='color:#a78bfa;font-size:1.5rem;font-weight:700;'>{_analyzed_cnt}</div>"
                    f"<div style='color:#a78bfa;font-size:0.8rem;font-weight:600;'>Properties</div>"
                    f"<div style='color:#a78bfa;font-size:0.78rem;'>({_analyzed_imgs} imgs)</div>"
                    f"<div style='color:#c4b5fd;font-size:0.75rem;margin-top:0.3rem;'>🔍 Analyzed & 🚫 Bad Quality</div>"
                    f"</div>"

                    f"<div style='background:rgba(245,158,11,0.15);border-radius:8px;padding:0.7rem;'>"
                    f"<div style='color:#f59e0b;font-size:1.5rem;font-weight:700;'>{_pending_cnt}</div>"
                    f"<div style='color:#f59e0b;font-size:0.8rem;font-weight:600;'>Properties</div>"
                    f"<div style='color:#f59e0b;font-size:0.78rem;'>({_pending_imgs} imgs)</div>"
                    f"<div style='color:#c4b5fd;font-size:0.75rem;margin-top:0.3rem;'>⏳ Pending</div>"
                    f"</div>"

                    f"</div></div>",
                    unsafe_allow_html=True
                )

                # Auto-load URLs when city is selected
                if True:
                    _parsed_urls  = []
                    _parse_errors = []
                    _seen_urls    = set()
                    # NEW: track source files and all property data for post-processing update
                    _source_files = []   # list of source current.json paths
                    _all_props    = []   # full property dicts (for optimized_yes.json)

                    for _sd in _scraper_output_root.iterdir():
                        if not _sd.is_dir():
                            continue
                        if _sel_scraper_src != "All Scrapers" and _sd.name != _sel_scraper_src:
                            continue
                        _cj = _sd / _sel_city_src / "current.json"
                        if not _cj.exists():
                            continue
                        try:
                            with open(_cj, "r", encoding="utf-8") as _cf:
                                _cprops = json.load(_cf)
                            _source_files.append(str(_cj))
                            for _prop in _cprops:
                                # ── Skip already optimised properties ─────────
                                if _prop.get("optimised") == "done" or _prop.get("optimized") == "done":
                                    continue
                                _all_props.append(_prop)
                                # Extract property_images (list of URL strings)
                                for _img in _prop.get("property_images", []):
                                    if isinstance(_img, str) and _img.startswith("http") and _img not in _seen_urls:
                                        _seen_urls.add(_img)
                                        _parsed_urls.append(_img)
                                # Extract floor plan images (list of strings OR dicts)
                                for _fp in _prop.get("floor_plans", []):
                                    if isinstance(_fp, str) and _fp.startswith("http") and _fp not in _seen_urls:
                                        _seen_urls.add(_fp)
                                        _parsed_urls.append(_fp)
                                    elif isinstance(_fp, dict):
                                        for _fk in ("image_url", "image_url_2d", "image_url_3d"):
                                            _furl = _fp.get(_fk)
                                            if _furl and isinstance(_furl, str) and _furl.startswith("http") and _furl not in _seen_urls:
                                                _seen_urls.add(_furl)
                                                _parsed_urls.append(_furl)
                                                break
                        except Exception as _je:
                            _parse_errors.append(f"{_cj}: {str(_je)}")

                    st.session_state["scraper_parsed_urls"]   = _parsed_urls
                    st.session_state["scraper_source_files"]  = _source_files
                    st.session_state["scraper_all_props"]     = _all_props
                    st.session_state["scraper_parsed_city"]   = _sel_city_src        # NEW
                    if _parse_errors:
                        st.warning(f"⚠️ {len(_parse_errors)} file(s) had errors: {'; '.join(_parse_errors[:3])}")
                if _parsed_urls:
                    scraper_json_urls = _parsed_urls
                    with st.expander(f"👁️ Preview URLs ({len(scraper_json_urls)} total)", expanded=False):
                        for _pu in scraper_json_urls[:20]:
                            st.caption(_pu)
                        if len(scraper_json_urls) > 20:
                            st.caption(f"… and {len(scraper_json_urls) - 20} more")

    process_disabled_t2 = True
    process_label_t2    = "👆 Input First"
    process_key_t2      = "process_ready_t2"

    if input_type == "📁 Local Folder Path" and folder_path_tab2 and os.path.isdir(folder_path_tab2):
        check_files = [f for f in os.listdir(folder_path_tab2)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.gif'))]
        if check_files:
            process_disabled_t2 = False
            process_label_t2    = f"🚀 Process {len(check_files)} Images from Folder"
            process_key_t2      = "process_folder_t2"
    elif input_type == "🔗 Direct URLs (one per line)" and url_list_tab2:
        process_disabled_t2 = False
        process_label_t2    = f"🚀 Download & Process {len(url_list_tab2)} URLs"
        process_key_t2      = "process_urls_t2"
    elif input_type == "🌐 API Endpoint" and st.session_state.api_fetched_urls:
        process_disabled_t2 = False
        process_label_t2    = f"🚀 Process {len(st.session_state.api_fetched_urls)} API Images"
        process_key_t2      = "process_api_t2"
    elif input_type == "📦 Scraper JSON Files" and scraper_json_urls:
        process_disabled_t2 = False
        _sc_name = st.session_state.get("img_opt_scraper_src", "scraper")
        _ci_name = st.session_state.get("img_opt_city_src", "")
        process_label_t2 = (
            f"🚀 Process {len(scraper_json_urls)} Images "
            f"— {_sc_name.title()} / {_ci_name.replace('_',' ').title()}"
        )
        process_key_t2 = "process_scraper_t2"

    if st.button(process_label_t2, type="primary", use_container_width=True,
                 disabled=process_disabled_t2 or st.session_state.processing,
                 key=process_key_t2):
        st.session_state.results           = []
        st.session_state.processed         = False
        st.session_state.processing        = True
        st.session_state.last_session_file = None
        session_start_t2                   = datetime.now()
        watermark_logo_t2  = load_watermark_logo()
        tr_settings_t2     = {'protect_center': protect_center,
                               'extra_margin': extra_margin, 'inpaint_radius': inpaint_radius}
        save_dir_t2 = custom_save_dir.strip() if custom_save_dir.strip() else URL_DOWNLOAD_DIR
        os.makedirs(save_dir_t2, exist_ok=True)
        image_paths_t2 = []

        if input_type == "📁 Local Folder Path":
            image_paths_t2 = [os.path.join(folder_path_tab2, f)
                               for f in os.listdir(folder_path_tab2)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.gif'))]
        else:
            if input_type == "🔗 Direct URLs (one per line)":
                urls_to_dl = url_list_tab2
                prefix_dl  = "url_img"
            elif input_type == "🌐 API Endpoint":
                urls_to_dl = st.session_state.api_fetched_urls
                prefix_dl  = "api_img"
            elif input_type == "📦 Scraper JSON Files":
                urls_to_dl = scraper_json_urls
                prefix_dl  = "scraper_img"
            else:
                urls_to_dl = []
                prefix_dl  = "img"
            dl_bar        = st.progress(0)
            dl_status     = st.empty()
            failed_dl     = []
            _fname_to_url = {}  # saved filename → original source URL
            for idx_dl, url_dl in enumerate(urls_to_dl):
                dl_status.info(f"📥 Downloading {idx_dl+1}/{len(urls_to_dl)}")
                try:
                    parsed_dl = urllib.parse.urlparse(url_dl)
                    ext_dl    = os.path.splitext(parsed_dl.path)[1].lower()
                    if ext_dl not in ALLOWED_EXTENSIONS:
                        ext_dl = '.jpg'
                    save_path_dl = os.path.join(save_dir_t2, f"{prefix_dl}_{idx_dl+1:04d}{ext_dl}")
                    req_dl = urllib.request.Request(url_dl, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req_dl, timeout=30) as resp_dl:
                        with open(save_path_dl, 'wb') as f_dl:
                            f_dl.write(resp_dl.read())
                    image_paths_t2.append(save_path_dl)
                    # Store mapping: saved filename → original URL
                    _fname_to_url[os.path.basename(save_path_dl)] = url_dl
                except Exception as e_dl:
                    failed_dl.append((url_dl, str(e_dl)))
                dl_bar.progress((idx_dl + 1) / len(urls_to_dl))
            dl_bar.empty(); dl_status.empty()
            # Save mapping in session state for post-processing writeback
            if input_type == "📦 Scraper JSON Files":
                st.session_state["scraper_fname_to_url"] = _fname_to_url
            if failed_dl:
                st.warning(f"⚠️ {len(failed_dl)} URLs failed.")
            if not image_paths_t2:
                st.error("No images downloaded.")
                st.session_state.processing = False
                st.stop()

        total_t2      = len(image_paths_t2)
        success_t2    = 0
        failed_cnt_t2 = 0
        progress_t2   = st.progress(0)
        status_t2     = st.empty()
        start_t2      = time.time()

        # ── Save processing state to URL so refresh can resume ────────────────
        _city_qp    = st.session_state.get("img_opt_city_src", "")
        _scraper_qp = st.session_state.get("img_opt_scraper_src", "")
        st.query_params["processing"] = "1"
        st.query_params["city"]       = _city_qp
        st.query_params["scraper"]    = _scraper_qp
        st.query_params["total"]      = str(total_t2)
        st.query_params["done"]       = "0"

        for i_t2, path_t2 in enumerate(image_paths_t2):
            fname_t2 = os.path.basename(path_t2)
            fsize_t2 = os.path.getsize(path_t2) if os.path.exists(path_t2) else 0
            status_t2.info(f"⏳ Processing {i_t2+1}/{total_t2}: {fname_t2}")
            try:
                result_t2 = process_single_image(
                    path_t2, fname_t2,
                    confidence_threshold / 100, quality_threshold,
                    fsize_t2, watermark_logo_t2,
                    enable_text_removal, tr_settings_t2,
                    enable_sr=enable_batch_sr,
                    sr_model_key=sr_model_key if enable_batch_sr else None,
                    sr_tile_size=tile_size,
                    sr_tile_overlap=tile_overlap
                )
                st.session_state.results.append(result_t2)
                if result_t2['status'] == 'success':
                    success_t2 += 1
                else:
                    failed_cnt_t2 += 1

                # ── Write current.json after EVERY image ──────────────────
                if input_type == "📦 Scraper JSON Files":
                    try:
                        _live_src_files    = st.session_state.get("scraper_source_files", [])
                        _live_fname_to_url = st.session_state.get("scraper_fname_to_url", {})

                        # Build url→r2 map from results so far
                        _live_url_to_r2 = {}
                        for _lr in st.session_state.results:
                            _lf  = _lr.get("filename", "")
                            _lop = _lr.get("output_path", "")
                            if _lr.get("status") != "success" or not _lf or not _lop:
                                continue
                            _lou = _live_fname_to_url.get(_lf)
                            if not _lou:
                                continue
                            _lq = _lr.get("quality_status", "Bad Quality")
                            if _lq == "Good Quality":
                                _live_url_to_r2[_lou] = os.path.basename(_lop)
                            else:
                                _live_url_to_r2[_lou] = None

                        # Update current.json for each source file
                        for _live_src in _live_src_files:
                            if not os.path.exists(_live_src):
                                continue
                            with open(_live_src, "r", encoding="utf-8") as _lrf:
                                _live_props = json.load(_lrf)
                            _live_changed = False
                            for _lp in _live_props:
                                _lopt = str(_lp.get("optimised") or _lp.get("optimized") or "no").strip().lower()
                                if _lopt in ("done", "analyzed"):
                                    continue
                                _l_imgs = _lp.get("property_images", [])
                                if not _l_imgs:
                                    continue
                                _lurls_processed = [
                                    u for u in _l_imgs
                                    if isinstance(u, str) and u in _live_url_to_r2
                                ]
                                if not _lurls_processed:
                                    continue

                                # Check ALL http URLs of this property are processed
                                # (skip if any image still pending — not yet downloaded)
                                _l_http_urls = [
                                    u for u in _l_imgs
                                    if isinstance(u, str) and u.startswith("http")
                                ]
                                _all_processed = all(
                                    u in _live_url_to_r2
                                    for u in _l_http_urls
                                )
                                if not _all_processed:
                                    continue  # wait — not all images done yet

                                _l_new_imgs = []
                                _l_has_r2   = False
                                for _lu in _l_imgs:
                                    if not isinstance(_lu, str):
                                        continue
                                    if _lu in _live_url_to_r2:
                                        _lv = _live_url_to_r2[_lu]
                                        if _lv is not None:
                                            _l_new_imgs.append(_lv)
                                            _l_has_r2 = True
                                        else:
                                            _l_new_imgs.append(_lu)
                                    else:
                                        _l_new_imgs.append(_lu)

                                _lp["property_images"] = _l_new_imgs
                                _lr2s = [u for u in _l_new_imgs if isinstance(u, str) and not u.startswith("http")]
                                if _lr2s:
                                    _lp["thumbnail"] = _lr2s[0]
                                _lp.pop("optimized", None)
                                _lp.pop("optimised", None)
                                _lp["optimised"] = "done" if _l_has_r2 else "analyzed"
                                _live_changed = True
                            if _live_changed:
                                with open(_live_src, "w", encoding="utf-8") as _lwf:
                                    json.dump(_live_props, _lwf, indent=2, ensure_ascii=False)
                    except Exception:
                        pass
                # ── END per-image writeback ───────────────────────────────

            except Exception as e_t2:

                failed_cnt_t2 += 1
                st.session_state.results.append({
                    "filename": fname_t2, "file_size": format_file_size(fsize_t2),
                    "category": "Error", "category_raw": "error",
                    "confidence": 0, "quality_status": "Error", "quality_score": 0,
                    "resolution": "N/A", "output_size": "N/A", "sr_info": "Error",
                    "text_removal": "Error", "text_removal_pct": 0,
                    "status": "failed", "error": str(e_t2)
                })
            progress_t2.progress((i_t2 + 1) / total_t2)

            # ── Update progress in URL query params ───────────────────────────
            st.query_params["done"] = str(i_t2 + 1)

            # ── PER-IMAGE WRITEBACK: resume support on browser refresh ────────
            if input_type == "📦 Scraper JSON Files":
                try:
                    _src_files_live    = st.session_state.get("scraper_source_files", [])
                    _fname_to_url_live = st.session_state.get("scraper_fname_to_url", {})

                    # Build orig_url → output mapping from results so far
                    _live_url_to_r2 = {}
                    for _r in st.session_state.results:
                        _fn = _r.get("filename", "")
                        _op = _r.get("output_path", "")
                        _ql = _r.get("quality_status", "Bad Quality")
                        if _r.get("status") != "success" or not _fn or not _op:
                            continue
                        _ou = _fname_to_url_live.get(_fn)
                        if not _ou:
                            continue
                        _rf = os.path.basename(_op)
                        _live_url_to_r2[_ou] = _rf if _ql == "Good Quality" else None

                    for _sp in _src_files_live:
                        if not os.path.exists(_sp):
                            continue
                        with open(_sp, "r", encoding="utf-8") as _rf2:
                            _sp_props = json.load(_rf2)
                        _sp_changed = False
                        for _pp in _sp_props:
                            # Skip already finalized
                            if _pp.get("optimised") in ("done", "analyzed"):
                                continue
                            _pp_imgs = _pp.get("property_images", [])
                            if not _pp_imgs:
                                continue
                            # Only write if ALL http images of this property are processed
                            _http_imgs = [u for u in _pp_imgs if isinstance(u, str) and u.startswith("http")]
                            if not _http_imgs:
                                continue
                            _all_done = all(_u in _live_url_to_r2 for _u in _http_imgs)
                            if not _all_done:
                                continue
                            # Build new images list
                            _ni     = []
                            _has_r2 = False
                            for _u2 in _pp_imgs:
                                if not isinstance(_u2, str):
                                    continue
                                _rv = _live_url_to_r2.get(_u2)
                                if _rv is not None:
                                    _ni.append(_rv)
                                    _has_r2 = True
                                else:
                                    _ni.append(_u2)
                            _pp["property_images"] = _ni
                            _r2i = [u for u in _ni if isinstance(u, str) and not u.startswith("http")]
                            if _r2i:
                                _pp["thumbnail"] = _r2i[0]
                            _pp.pop("optimized", None)
                            _pp.pop("optimised", None)
                            _pp["optimised"] = "done" if _has_r2 else "analyzed"
                            _sp_changed = True
                        if _sp_changed:
                            with open(_sp, "w", encoding="utf-8") as _wf2:
                                json.dump(_sp_props, _wf2, indent=2, ensure_ascii=False)
                except Exception:
                    pass
            # ── END PER-IMAGE WRITEBACK ───────────────────────────────────────

        processing_time_t2 = time.time() - start_t2

        processing_time_t2 = time.time() - start_t2
        update_statistics(st.session_state.results)
        add_upload_record(st.session_state.results)
        session_file_t2 = save_session_report(st.session_state.results, processing_time_t2, session_start_t2)
        st.session_state.last_session_file = session_file_t2
        st.session_state.processed  = True
        st.session_state.processing = False
        status_t2.empty(); progress_t2.empty()

        # ── Write optimised results back into current.json ────────────────────
        # This runs after ALL images done AND also called per-image for resume support
        if input_type == "📦 Scraper JSON Files":
            try:
                _src_files    = st.session_state.get("scraper_source_files", [])
                _all_props    = st.session_state.get("scraper_all_props", [])
                _fname_to_url = st.session_state.get("scraper_fname_to_url", {})

                if _src_files and _all_props and _fname_to_url:

                    # ── Step 1: Build original_url → R2 CDN URL ───────────────
                    # filename → original_url (from download mapping)
                    # original_url → R2 url (from result output_path + category)
                    _orig_url_to_r2 = {}
                    for _res in st.session_state.results:
                        _fname    = _res.get("filename", "")
                        _out_path = _res.get("output_path", "")
                        _category = _res.get("category_raw", "gallery")
                        _quality  = _res.get("quality_status", "Bad Quality")

                        if _res.get("status") != "success":
                            continue
                        if not _fname or not _out_path:
                            continue

                        _orig_url = _fname_to_url.get(_fname)
                        if not _orig_url:
                            continue

                        _r2_fname = os.path.basename(_out_path)
                        _quality  = _res.get("quality_status", "Bad Quality")

                        if _quality == "Good Quality":
                            # ✅ Good Quality → store filename (uploaded to R2)
                            _orig_url_to_r2[_orig_url] = _r2_fname
                        else:
                            # ❌ Bad Quality → store None (processed but NOT uploaded)
                            _orig_url_to_r2[_orig_url] = None
                    # ── Step 2: Update each current.json in-place ─────────────
                    _total_done    = 0
                    _total_skipped = 0

                    for _src_path in _src_files:
                        if not os.path.exists(_src_path):
                            continue

                        with open(_src_path, "r", encoding="utf-8") as _rf:
                            _src_props = json.load(_rf)

                        _changed = False

                        # ── Auto-fix: replace any existing full R2 URLs with filename only ──
                        for _prop in _src_props:
                            _imgs = _prop.get("property_images", [])
                            _fixed_imgs = []
                            _img_changed = False
                            for _img in _imgs:
                                if isinstance(_img, str) and "r2.dev" in _img and "/img/" in _img:
                                    # Extract just the filename from full R2 URL
                                    _fixed_imgs.append(_img.split("/")[-1])
                                    _img_changed = True
                                else:
                                    _fixed_imgs.append(_img)
                            if _img_changed:
                                _prop["property_images"] = _fixed_imgs
                                # Fix thumbnail too
                                _thumb = _prop.get("thumbnail", "")
                                if isinstance(_thumb, str) and "r2.dev" in _thumb and "/img/" in _thumb:
                                    _prop["thumbnail"] = _thumb.split("/")[-1]
                                _changed = True

                        for _prop in _src_props:

                            # Skip already optimised or analyzed — never reprocess
                            if _prop.get("optimised") in ("done", "analyzed") or _prop.get("optimized") == "done":
                                _total_skipped += 1
                                continue

                            _orig_imgs = _prop.get("property_images", [])
                            if not _orig_imgs:
                                continue

                            # Check if ANY image was successfully processed
                            _orig_url_set = set(_fname_to_url.values())
                            _any_processed = any(
                                _u in _orig_url_set
                                for _u in _orig_imgs
                                if isinstance(_u, str)
                            )
                            if not _any_processed:
                                continue
                            # Build new images list
                            _new_imgs      = []
                            _has_r2_upload = False

                            for _orig_url in _orig_imgs:
                                if not isinstance(_orig_url, str):
                                    continue
                                # Check if this URL was processed
                                if _orig_url in _orig_url_to_r2:
                                    _r2_val = _orig_url_to_r2[_orig_url]
                                    if _r2_val is not None:
                                        # ✅ Good Quality → store filename only
                                        _new_imgs.append(_r2_val)
                                        _has_r2_upload = True
                                    else:
                                        # ❌ Bad Quality → keep original URL
                                        _new_imgs.append(_orig_url)
                                else:
                                    # Not processed → keep original URL
                                    _new_imgs.append(_orig_url)

                            _prop["property_images"] = _new_imgs

                            # Update thumbnail → first R2 filename if available
                            _r2_imgs = [u for u in _new_imgs if isinstance(u, str) and not u.startswith("http")]
                            if _r2_imgs:
                                _prop["thumbnail"] = _r2_imgs[0]

                            # Always remove both spellings first
                            _prop.pop("optimized", None)
                            _prop.pop("optimised", None)

                            # Set status based on result:
                            # "done"     = at least 1 Good Quality image uploaded to Cloudflare R2
                            # "analyzed" = processed but ALL images were Bad Quality, nothing uploaded
                            _prop["optimised"] = "done" if _has_r2_upload else "analyzed"
                            _changed    = True
                            _total_done += 1
                    # ── Clean up ALL props in file ────────────────────────────
                    for _prop in _src_props:
                        # Always remove American spelling
                        if "optimized" in _prop:
                            _prop.pop("optimized", None)
                            _changed = True
                        # Properties with no images → default "no"
                        _imgs = _prop.get("property_images", [])
                        if not _imgs and _prop.get("optimised") not in ("done", "analyzed"):
                            _prop["optimised"] = "no"
                            _changed = True

                    # Write back ONLY if something changed
                    if _changed:
                        with open(_src_path, "w", encoding="utf-8") as _wf:
                            json.dump(
                                _src_props, _wf,
                                indent=2, ensure_ascii=False
                            )
                        st.success(
                            f"✅ `current.json` updated — "
                            f"`{os.path.abspath(_src_path)}`"
                        )
                    # ── Count analyzed and bad quality from this session ──
                    _session_bad_imgs = len([
                        r for r in st.session_state.results
                        if r.get("quality_status") == "Bad Quality"
                        and r.get("status") == "success"
                    ])
                    _session_analyzed_props = len([
                        r for r in st.session_state.results
                        if r.get("quality_status") == "Bad Quality"
                        and r.get("status") == "success"
                    ])
                    # ── Store in session state so dashboard can read it ───
                    st.session_state["last_analyzed_imgs"]   = _session_bad_imgs
                    st.session_state["last_bad_quality_imgs"] = _session_bad_imgs
                    st.success(
                        f"🎉 **{_total_done}** properties written back to `current.json` "
                        f"with `optimised: done`"
                    )
                    st.info(
                        f"🔍 **Analyzed:** {_session_analyzed_props} properties | "
                        f"🚫 **Bad Quality Images:** {_session_bad_imgs} images"
                    )
                    if _total_skipped:
                        st.info(
                            f"⏭️ **{_total_skipped}** properties already had "
                            f"`optimised: done` — skipped"
                        )

            except Exception as _opt_err:
                st.warning(f"⚠️ current.json update failed: {_opt_err}")
                import traceback
                st.code(traceback.format_exc())
        # ── END optimised writeback ───────────────────────────────────────────
    
        # ── END optimized save ────────────────────────────────────────────────
        # ── Clear URL query params — processing complete ──────────────────────
        st.query_params.clear()

        st.balloons()
        st.success(f"🎉 Done! ✅ {success_t2} success | ❌ {failed_cnt_t2} failed")
        _r2_ph_t2  = st.empty()
        _r2_res_t2 = auto_upload_to_r2(st.session_state.results, _r2_ph_t2)
        _r2_ph_t2.empty()
        if _r2_res_t2["failed"] == 0 and _r2_res_t2["uploaded"] > 0:
            st.success(f"☁️ R2 Upload Complete — {_r2_res_t2['uploaded']} file(s) sent to R2.")
        elif _r2_res_t2["uploaded"] > 0:
            st.warning(f"☁️ R2 Partial — ✅ {_r2_res_t2['uploaded']} uploaded | ❌ {_r2_res_t2['failed']} failed")
        else:
            st.error(f"☁️ R2 Upload Failed — {'; '.join(_r2_res_t2['errors'][:3])}")
        time.sleep(2)
        st.rerun()

# ── TAB 3: Super-Resolution (Single Image) ────────────────────────────────────
with input_tab3:
    st.markdown("""
<div style='background: rgba(30, 15, 50, 0.6); padding: 1.5rem; border-radius: 12px;
     border: 1px solid rgba(139, 92, 246, 0.3); margin-bottom: 1.5rem;'>
    <h2 style='text-align: center; color: #8b5cf6; margin: 0 0 0.5rem 0;'>🔬 PropVision AI — Super-Resolution</h2>
    <p style='text-align: center; color: #c4b5fd; margin: 0; font-size: 0.95rem;'>
        Transform low-resolution property images into sharp, photorealistic high-resolution output
        using Real-ESRGAN deep learning.
    </p>
</div>
""", unsafe_allow_html=True)

    sr_c1, sr_c2, sr_c3 = st.columns(3)
    for _col, _icon, _title, _body in zip(
        [sr_c1, sr_c2, sr_c3], ["🏠", "🏢", "🖼️"],
        ["Interior / Exterior Photos", "Gentle Upscale", "Renders / CG / Floorplans"],
        ["Use <b>Real-ESRGAN ×4</b> — restores natural textures, sharpens edges without artifacts.",
         "Use <b>Real-ESRGAN ×2</b> — doubles resolution while keeping the image looking natural.",
         "Use <b>Anime ×4</b> — best for CGI renders, floorplan diagrams, and illustrations."]):
        with _col:
            st.markdown(f"""
<div style='background: rgba(139,92,246,0.1); border: 1px solid rgba(139,92,246,0.3);
     border-radius: 12px; padding: 1rem; margin-bottom: 1rem;'>
  <div style='font-size:1.8rem; margin-bottom:0.5rem'>{_icon}</div>
  <div style='font-weight:600; color:#e0e7ff; margin-bottom:0.4rem'>{_title}</div>
  <div style='color:#c4b5fd; font-size:0.84rem; line-height:1.5'>{_body}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    sr_uploaded = st.file_uploader(
        "Drop your low-resolution property image here",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        key="sr_uploader"
    )

    if sr_uploaded is None:
        st.markdown("""
<div class='status-bar'>
  <b>How it works:</b> Upload any property image → choose a model in the sidebar →
  click <em>✦ Enhance Image</em> → download the sharp, high-resolution result.
</div>
""", unsafe_allow_html=True)
    else:
        sr_input_img = Image.open(sr_uploaded).convert("RGB")
        sr_iw, sr_ih = sr_input_img.size

        sr_col_in, sr_col_out = st.columns(2, gap="large")

        with sr_col_in:
            st.markdown("""
<div class="img-panel">
  <div class="img-panel-header">
    <span class="img-panel-badge badge-input">Input</span>
    &nbsp;&nbsp;Low-resolution source
  </div>""", unsafe_allow_html=True)
            st.image(sr_input_img, use_container_width=True)
            st.markdown(f"""
  <div class="img-panel-body">
    <div class="stats-row">
      <div class="stat-chip">Size: <b>{sr_iw} × {sr_ih}</b></div>
      <div class="stat-chip">Format: <b>{sr_uploaded.type.split("/")[1].upper()}</b></div>
      <div class="stat-chip">File: <b>{sr_uploaded.size // 1024} KB</b></div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        with sr_col_out:
            sr_out_ph = st.empty()
            sr_out_ph.markdown("""
<div class="img-panel" style="min-height:320px;display:flex;
  align-items:center;justify-content:center;">
  <div style="text-align:center;color:#7a7d8a;padding:3rem">
    <div style="font-size:2.5rem;margin-bottom:0.8rem">✨</div>
    <div style="font-weight:500">Enhanced image will appear here</div>
    <div style="font-size:0.82rem;margin-top:0.3rem">
      Click <em>✦ Enhance Image</em> below</div>
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        sr_enhance_btn = st.button("✦  Enhance Image", use_container_width=True, key="sr_enhance_btn")

        def render_sr_output(res_img, res_scale, res_bytes, res_stem):
            sw, sh = res_img.size
            with sr_col_out:
                sr_out_ph.empty()
                st.markdown("""
<div class="img-panel">
  <div class="img-panel-header">
    <span class="img-panel-badge badge-output">Enhanced</span>
    &nbsp;&nbsp;Real-ESRGAN output
  </div>""", unsafe_allow_html=True)
                st.image(res_img, use_container_width=True)
                st.markdown(f"""
  <div class="img-panel-body">
    <div class="stats-row">
      <div class="stat-chip">Size: <b>{sw} × {sh}</b></div>
      <div class="stat-chip">Scale: <b>{res_scale}×</b></div>
      <div class="stat-chip">Model: <b>Real-ESRGAN</b></div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
            st.markdown(f"""
<div class="status-bar" style="border-left-color:#10b981;">
  ✅ <b>Enhancement complete!</b>&nbsp;
  Output: {sw} × {sh} px &nbsp;·&nbsp; {res_scale}× upscale
</div>""", unsafe_allow_html=True)
            st.download_button(
                "⬇  Download Enhanced Image",
                data=res_bytes,
                file_name=f"{res_stem}_enhanced_{res_scale}x.png",
                mime="image/png",
                use_container_width=True,
                key="sr_download_btn"
            )

        if sr_enhance_btn:
            with st.spinner("Loading Real-ESRGAN model…"):
                try:
                    _sr_model, _sr_scale, _sr_dev = load_sr_model(sr_model_key)
                except Exception as _e:
                    st.error(f"Model load failed: {_e}")
                    st.stop()
            try:
                _pb = st.progress(0.3, text="Running enhancement…")
                _sr_result = run_sr(_sr_model, _sr_scale, _sr_dev, sr_input_img,
                                    tile_size, tile_overlap)
                _pb.progress(1.0); _pb.empty()
                _sr_bytes = image_to_bytes(_sr_result)
                _sr_stem  = Path(sr_uploaded.name).stem
                st.session_state.sr_img            = _sr_result
                st.session_state.sr_bytes          = _sr_bytes
                st.session_state.sr_scale          = _sr_scale
                st.session_state.sr_model_key_used = sr_model_key
                st.session_state.sr_stem           = _sr_stem
                render_sr_output(_sr_result, _sr_scale, _sr_bytes, _sr_stem)
            except Exception as _e:
                _pb.empty()
                st.error(f"Enhancement failed: {_e}")
                import traceback
                st.code(traceback.format_exc())

        elif (st.session_state.sr_img is not None and
              st.session_state.sr_model_key_used == sr_model_key):
            render_sr_output(
                st.session_state.sr_img,
                st.session_state.sr_scale,
                st.session_state.sr_bytes,
                st.session_state.sr_stem
            )

# ===================== RESULTS =====================
if st.session_state.processed and st.session_state.results:
    df = pd.DataFrame(st.session_state.results)

    st.markdown("""
<div style='background: rgba(30, 15, 50, 0.6); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(139, 92, 246, 0.3); margin-bottom: 2rem;'>
    <h2 style='text-align: center; color: #8b5cf6; margin: 0;'>📊 Current Session Analytics</h2>
</div>
""", unsafe_allow_html=True)

    total        = len(df)
    successful   = len(df[df['status'] == 'success'])
    property_img = len(df[df['category_raw'] == 'gallery'])
    floor_img    = len(df[df['category_raw'] == 'floorplan'])
    master_img   = len(df[df['category_raw'] == 'masterplan'])
    rejected_img = len(df[df['category_raw'] == 'rejected'])
    good_qual    = len(df[df['quality_status'] == 'Good Quality'])
    bad_qual     = len(df[df['quality_status'] == 'Bad Quality'])
    df_success   = df[df['status'] == 'success']
    avg_conf         = df_success['confidence'].mean()       if len(df_success) > 0 else 0
    avg_qual         = df_success['quality_score'].mean()    if len(df_success) > 0 else 0
    avg_text_removal = df_success['text_removal_pct'].mean() if len(df_success) > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🗺️ Master Plans",  master_img)
        st.metric("✅ Successful",     successful, f"{successful/total*100:.1f}%")
    with col2:
        st.metric("🖼️ Gallery Images", property_img)
        st.metric("❌ Rejected",        rejected_img)
    with col3:
        st.metric("📐 Floor Plans",   floor_img)
        st.metric("📁 Total Images",  total)
    with col4:
        st.metric("🌟 Good Quality",  good_qual, f"{good_qual/total*100:.1f}%")
        st.metric("⚠️ Bad Quality",   bad_qual,  f"{bad_qual/total*100:.1f}%")

    col5, col6, col7, col8 = st.columns(4)
    with col5: st.metric("🎯 Avg Confidence",  f"{avg_conf:.1f}%")
    with col6: st.metric("🔍 Avg Blur Score",  f"{avg_qual:.1f}")
    with col7:
        if enable_text_removal and ocr_reader:
            st.metric("🧹 Avg Text Removed", f"{avg_text_removal:.1f}%")
        else:
            st.metric("🧹 Text Removal", "Disabled")
    with col8: st.metric("📊 Success Rate", f"{successful/total*100:.1f}%")

    if st.session_state.last_session_file and not str(st.session_state.last_session_file).startswith("ERROR"):
        try:
            with open(st.session_state.last_session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            st.markdown("""
<div style='background: rgba(16, 185, 129, 0.1); padding: 1.5rem; border-radius: 12px;
     border: 1px solid rgba(16, 185, 129, 0.4); margin: 2rem 0;'>
    <h2 style='text-align: center; color: #10b981; margin: 0;'>📋 Auto-Saved Session Report</h2>
</div>
""", unsafe_allow_html=True)
            s = session_data.get("overall_summary", {})
            r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)
            r2c1.metric("📁 Total",     s.get("total_images",    0))
            r2c2.metric("✅ Processed", s.get("total_processed", 0))
            r2c3.metric("❌ Failed",    s.get("total_failed",    0))
            r2c4.metric("🌟 Good",      s.get("good_quality",    0))
            r2c5.metric("⚠️ Bad",       s.get("bad_quality",     0))
            st.markdown("---")
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                with open(st.session_state.last_session_file, 'rb') as f_dl:
                    st.download_button("📥 Download Session Report (JSON)", data=f_dl.read(),
                                       file_name=os.path.basename(st.session_state.last_session_file),
                                       mime="application/json", use_container_width=True)
            with dl_col2:
                if os.path.exists(ALL_SESSIONS_FILE):
                    with open(ALL_SESSIONS_FILE, 'rb') as f_all:
                        st.download_button("📥 Download All Sessions Log (JSON)", data=f_all.read(),
                                           file_name="all_sessions_log.json",
                                           mime="application/json", use_container_width=True)
        except Exception as e_rep:
            st.warning(f"Could not display session report: {e_rep}")

    try:
        import plotly.express as px
        layout_extra = {
            'paper_bgcolor': 'rgba(30, 15, 50, 0.5)',
            'plot_bgcolor':  'rgba(20, 10, 40, 0.3)',
            'font': {'color': '#e0e7ff'}
        }
        col1, col2 = st.columns(2)
        with col1:
            cat_counts = df['category'].value_counts().reset_index()
            cat_counts.columns = ['Category', 'Count']
            fig1 = px.bar(cat_counts, x='Category', y='Count', title='📊 Category Distribution',
                          color='Category', color_discrete_sequence=['#8b5cf6', '#7c3aed', '#6d28d9', '#5b21b6'])
            fig1.update_layout(**layout_extra, height=400)
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            qual_counts = df['quality_status'].value_counts().reset_index()
            qual_counts.columns = ['Quality', 'Count']
            fig2 = px.pie(qual_counts, names='Quality', values='Count', title='🎯 Quality Distribution',
                          color_discrete_sequence=['#10b981', '#ef4444', '#6b7280'])
            fig2.update_layout(**layout_extra, height=400)
            st.plotly_chart(fig2, use_container_width=True)
    except ImportError:
        st.warning("⚠️ Install plotly: `pip install plotly`")

    with st.expander("🌐 Send Data to Server", expanded=False):
        srv_col1, srv_col2 = st.columns(2)
        with srv_col1:
            server_url  = st.text_input("🌐 Server Endpoint URL", key="srv_url")
            api_key_srv = st.text_input("🔑 API Key (optional)", type="password", key="srv_apikey")
        with srv_col2:
            extra_header_name  = st.text_input("➕ Extra Header Name",  key="srv_hname")
            extra_header_value = st.text_input("➕ Extra Header Value", key="srv_hval")
            send_only_success  = st.checkbox("✅ Send only successful", value=True, key="srv_success")
        send_disabled = not bool(server_url and server_url.startswith("http"))
        if st.button("🚀 SEND TO SERVER", use_container_width=True,
                     disabled=send_disabled, key="btn_send_server"):
            payload_results = ([r for r in st.session_state.results if r.get('status') == 'success']
                               if send_only_success else st.session_state.results)
            extra_headers = {}
            if extra_header_name and extra_header_value:
                extra_headers[extra_header_name] = extra_header_value
            with st.spinner(f"📡 Sending {len(payload_results)} records…"):
                ok, msg = send_results_to_server(payload_results, server_url, api_key_srv, extra_headers)
            if ok: st.success(msg); st.balloons()
            else:  st.error(msg)

    st.markdown("""
<div style='background: rgba(30, 15, 50, 0.6); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(139, 92, 246, 0.3); margin: 2rem 0;'>
    <h2 style='text-align: center; color: #8b5cf6; margin: 0;'>📋 Detailed Results Table</h2>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        cat_filter = st.multiselect("🏷️ Filter by Category",
                                    options=df['category'].unique().tolist(),
                                    default=df['category'].unique().tolist())
    with col2:
        qual_filter = st.multiselect("🌟 Filter by Quality",
                                     options=df['quality_status'].unique().tolist(),
                                     default=df['quality_status'].unique().tolist())
    with col3:
        status_filter = st.multiselect("📊 Filter by Status",
                                       options=df['status'].unique().tolist(),
                                       default=df['status'].unique().tolist())

    filtered_df = df[
        (df['category'].isin(cat_filter)) &
        (df['quality_status'].isin(qual_filter)) &
        (df['status'].isin(status_filter))
    ]
    st.info(f"📋 Showing **{len(filtered_df)}** of **{total}** images")

    desired_columns = ['filename', 'file_size', 'category', 'confidence', 'quality_status',
                       'quality_score', 'resolution', 'sr_info', 'output_size',
                       'text_removal', 'status', 'error']
    display_columns = [col for col in desired_columns if col in filtered_df.columns]
    st.dataframe(filtered_df[display_columns], use_container_width=True, height=400)

    st.markdown("""
<div style='background: rgba(30, 15, 50, 0.6); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(139, 92, 246, 0.3); margin: 2rem 0;'>
    <h2 style='text-align: center; color: #8b5cf6; margin: 0;'>💾 Download Reports & Images</h2>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with col1:
        st.download_button("📄 CSV Report", filtered_df.to_csv(index=False).encode('utf-8'),
                           f"homes247_{timestamp}.csv", "text/csv", use_container_width=True)
    with col2:
        st.download_button("📋 JSON Report", filtered_df.to_json(orient='records', indent=2),
                           f"homes247_{timestamp}.json", "application/json", use_container_width=True)
    with col3:
        try:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='Results')
                pd.DataFrame({
                    'Metric': ['Total', 'Successful', 'Failed', 'Gallery', 'Floor Plans',
                               'Master Plans', 'Rejected', 'Good Quality', 'Bad Quality'],
                    'Value': [total, successful, total - successful, property_img,
                              floor_img, master_img, rejected_img, good_qual, bad_qual]
                }).to_excel(writer, index=False, sheet_name='Summary')
            st.download_button("📊 Excel Report", output.getvalue(),
                               f"homes247_{timestamp}.xlsx", "application/vnd.ms-excel",
                               use_container_width=True)
        except:
            st.info("📦 Install openpyxl: `pip install openpyxl`")
    with col4:
        with st.spinner("📦 Creating ZIP..."):
            zip_buffer = create_download_zip(st.session_state.results, timestamp)
            if zip_buffer:
                st.download_button("📦 Complete ZIP", zip_buffer.getvalue(),
                                   f"homes247_complete_{timestamp}.zip", "application/zip",
                                   use_container_width=True)

else:
    st.markdown("""
<div style='text-align: center; padding: 3rem; background: rgba(30, 15, 50, 0.6); border-radius: 15px; border: 1px solid rgba(139, 92, 246, 0.3);'>
    <h2 style='color: #8b5cf6; margin: 0 0 2rem 0;'>👋 Welcome to Homes247 AI Dashboard</h2>
    <div style='text-align: left; background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 10px;'>
        <h3 style='color: #8b5cf6; margin: 0 0 1rem 0;'>🚀 Quick Start:</h3>
        <ol style='color: #c4b5fd; margin: 0; padding-left: 1.5rem;'>
            <li>Upload your logo in sidebar (watermark)</li>
            <li>Enable AI Text Removal (optional)</li>
            <li>Set <b>Blur Threshold</b> in sidebar (default 500 — adjust to your images)</li>
            <li>Enable SR in sidebar to upscale after text removal</li>
            <li><b>Tab 1</b>: Upload files &nbsp; <b>OR</b> &nbsp; <b>Tab 2</b>: Paste image URLs / folder path</li>
            <li><b>Tab 3</b>: Use Super-Resolution to enhance any single low-res image</li>
            <li>Pipeline order: <b>Text Removal → SR → Resize → Watermark → Save</b></li>
            <li>Download reports or send to your server</li>
        </ol>
    </div>
</div>
""", unsafe_allow_html=True)

# ===================== FOOTER =====================
st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); border-radius: 15px; margin-top: 3rem;'>
    <h2 style='margin: 0; color: white; font-size: 2rem;'>🏠 HOMES247</h2>
    <p style='margin: 0.5rem 0 0 0; color: #e9d5ff;'>India's Favourite Property Portal</p>
    <p style='margin: 1rem 0 0 0; color: #c4b5fd; font-size: 0.9rem;'>Version 3.1 Professional | Developed by Middi Yogananda Reddy</p>
    <p style='margin: 0.5rem 0 0 0; color: #c4b5fd; font-size: 0.8rem;'>© 2026 Homes247. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)    