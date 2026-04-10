"""
Homes247 Premium Real Estate Image Processing Dashboard
Professional Streamlit Application with Modern Dark Theme UI
Version: 2.0 Professional Edition - UNLIMITED BULK PROCESSING
WITH PERSISTENT IMAGE COUNT TRACKING
INTEGRATED AI IMAGE CLEANER FOR LOGO/TEXT/WATERMARK REMOVAL
"""

# Suppress TensorFlow warnings FIRST (before any imports)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import shutil
import base64
from pathlib import Path
from PIL import Image
import cv2
import time
import easyocr
import io

# Suppress TensorFlow deprecation warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Import existing pipeline
from predict import predict_image

# ===================== CONFIGURATION =====================
BASE_DIR = r"C:\Users\Homes247\Desktop\Bulk_image"
OUTPUT_DIR = os.path.join(BASE_DIR, "streamlit_output")
UPLOAD_DIR = os.path.join(OUTPUT_DIR, "uploads")
STATS_FILE = os.path.join(BASE_DIR, "processing_statistics.json")

CATEGORY_MAPPING = {
    "floorplan": "Floor Plan Images",
    "masterplan": "Master Plan Images",
    "gallery": "Gallery Images",
    "rejected": "Others"
}

CATEGORY_SIZES = {
    "floorplan": (1500, 1500),
    "masterplan": (1640, 860),
    "gallery": (820, 430),
    "rejected": None
}

# ===================== INTEGRATED CLEANING CONFIG =====================
CATEGORY_OPTIONS = ["Master Plan", "Floor Plan", "Gallery"]  # Maps to masterplan, floorplan, gallery

KEYWORDS = {
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

SETTINGS = {
    "Master Plan": {"corner_pct": 0.20, "edge_pct": 0.12, "ocr_margin": 0.25},
    "Floor Plan":  {"corner_pct": 0.15, "edge_pct": 0.10, "ocr_margin": 0.20},
    "Gallery":     {"corner_pct": 0.12, "edge_pct": 0.08, "ocr_margin": 0.15},
}

# Cached OCR Reader
@st.cache_resource(show_spinner="Loading AI OCR engine…")
def load_ocr():
    try:
        return easyocr.Reader(['en'], gpu=False)
    except Exception as e:
        st.error(f"EasyOCR load error: {e}")
        return None

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ===================== STATISTICS MANAGEMENT =====================

def load_statistics():
    """Load statistics from JSON file"""
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    
    # Default statistics structure
    return {
        "total_processed": 0,
        "floorplan_count": 0,
        "masterplan_count": 0,
        "gallery_count": 0,
        "rejected_count": 0,
        "good_quality_count": 0,
        "bad_quality_count": 0,
        "first_upload_date": None,
        "last_upload_date": None,
        "total_sessions": 0,
        "processing_history": []
    }

def save_statistics(stats):
    """Save statistics to JSON file"""
    try:
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving statistics: {e}")
        return False

def update_statistics(results):
    """Update statistics with new processing results"""
    stats = load_statistics()
    
    # Count new images by category
    new_floorplan = len([r for r in results if r.get('category_raw') == 'floorplan' and r.get('status') == 'success'])
    new_masterplan = len([r for r in results if r.get('category_raw') == 'masterplan' and r.get('status') == 'success'])
    new_gallery = len([r for r in results if r.get('category_raw') == 'gallery' and r.get('status') == 'success'])
    new_rejected = len([r for r in results if r.get('category_raw') == 'rejected' and r.get('status') == 'success'])
    
    # Count by quality
    new_good = len([r for r in results if r.get('quality_status') == 'Good Quality' and r.get('status') == 'success'])
    new_bad = len([r for r in results if r.get('quality_status') == 'Bad Quality' and r.get('status') == 'success'])
    
    total_new = new_floorplan + new_masterplan + new_gallery + new_rejected
    
    # Update cumulative counts
    stats['total_processed'] += total_new
    stats['floorplan_count'] += new_floorplan
    stats['masterplan_count'] += new_masterplan
    stats['gallery_count'] += new_gallery
    stats['rejected_count'] += new_rejected
    stats['good_quality_count'] += new_good
    stats['bad_quality_count'] += new_bad
    
    # Update dates
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if stats['first_upload_date'] is None:
        stats['first_upload_date'] = current_date
    stats['last_upload_date'] = current_date
    
    # Increment sessions
    stats['total_sessions'] += 1
    
    # Add to history
    stats['processing_history'].append({
        'date': current_date,
        'total': total_new,
        'floorplan': new_floorplan,
        'masterplan': new_masterplan,
        'gallery': new_gallery,
        'rejected': new_rejected,
        'good_quality': new_good,
        'bad_quality': new_bad
    })
    
    # Keep only last 100 history entries
    if len(stats['processing_history']) > 100:
        stats['processing_history'] = stats['processing_history'][-100:]
    
    save_statistics(stats)
    return stats

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Homes247 - AI Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== HELPER FUNCTIONS =====================

def get_base64_image(image_path):
    """Convert local image to base64"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

def format_file_size(size_bytes):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def normalize_category(label: str):
    """ENHANCED: Normalize model output with strict rules to prevent misclassification"""
    if not label:
        return "rejected"

    # Convert to lowercase and remove special characters for matching
    l = label.lower().strip()
    original_label = l  # Keep original for debugging
    l = l.replace(" ", "").replace("_", "").replace("-", "")

    # STRICT PRIORITY MATCHING - Order matters!
    
    # 1. FLOOR PLAN - Check first with specific keywords
    floor_keywords = ['floor', 'floorplan', 'unit', 'apartment', 'bedroom', 'bhk', 'layout']
    if any(keyword in l for keyword in floor_keywords):
        # Double-check it's not a master plan
        master_keywords = ['master', 'site', 'siteplan', 'location', 'compound']
        if not any(keyword in l for keyword in master_keywords):
            return "floorplan"
    
    # 2. MASTER PLAN - Check with specific keywords
    master_keywords = ['master', 'masterplan', 'site', 'siteplan', 'location', 'compound', 'complex', 'development']
    if any(keyword in l for keyword in master_keywords):
        return "masterplan"
    
    # 3. GALLERY - Check for property/exterior/interior images
    gallery_keywords = ['gallery', 'photo', 'image', 'property', 'exterior', 'interior', 'view', 'render', 'elevation']
    if any(keyword in l for keyword in gallery_keywords):
        return "gallery"

    # 4. If none matched, it's rejected
    return "rejected"

# ===================== PREMIUM DARK THEME CSS =====================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Main Background - Dark Purple Gradient */
    .stApp {
        background: linear-gradient(135deg, #1a0033 0%, #2d1b4e 25%, #1f0d3d 50%, #0f0520 100%);
        background-attachment: fixed;
    }
    
    /* Header with Logo and Branding */
    .main-header {
        background: linear-gradient(135deg, #8B1538 0%, #9d2449 50%, #7d1230 100%);
        padding: 2rem 3rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(139, 21, 56, 0.4);
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .header-content {
        display: flex;
        align-items: center;
        gap: 2rem;
        z-index: 2;
    }
    
    .logo-container {
        width: 80px;
        height: 80px;
        background: white;
        border-radius: 15px;
        padding: 10px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.3);
    }
    
    .header-text h1 {
        color: white;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
    }
    
    .header-text p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Statistics Banner - NEW */
    .stats-banner {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.15) 100%);
        border: 2px solid rgba(16, 185, 129, 0.5);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.2);
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .stat-item {
        background: rgba(30, 15, 50, 0.6);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }
    
    .stat-value {
        color: #10b981;
        font-size: 2rem;
        font-weight: 800;
        margin: 0.5rem 0;
        text-shadow: 0 0 20px rgba(16, 185, 129, 0.5);
    }
    
    .stat-label {
        color: #c4b5fd;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0;
    }
    
    /* Sidebar Styling - Dark Theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0f2e 0%, #0d0520 100%);
        border-right: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    [data-testid="stSidebar"] h3 {
        color: #a78bfa;
        font-weight: 700;
        font-size: 1.3rem;
        padding: 1rem 0;
        border-bottom: 2px solid rgba(139, 92, 246, 0.3);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e7ff;
    }
    
    /* Section Headers - Neon Glow Effect */
    .section-header {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(124, 58, 237, 0.1) 100%);
        border-left: 5px solid #8b5cf6;
        padding: 1.5rem 2rem;
        margin: 2rem 0 1rem 0;
        border-radius: 12px;
        box-shadow: 0 0 30px rgba(139, 92, 246, 0.2);
    }
    
    .section-header h2 {
        color: #c4b5fd;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 0 10px rgba(139, 92, 246, 0.5);
    }
    
    /* Metric Cards - Glassmorphism Dark */
    div[data-testid="stMetric"] {
        background: rgba(30, 15, 50, 0.6);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(139, 92, 246, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(139, 92, 246, 0.4);
        border-color: rgba(139, 92, 246, 0.6);
    }
    
    div[data-testid="stMetricLabel"] {
        color: #c4b5fd !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        text-shadow: 0 0 20px rgba(139, 92, 246, 0.5);
    }
    
    div[data-testid="stMetricDelta"] {
        color: #86efac !important;
    }
    
    /* Buttons - Premium Gradient */
    .stButton>button {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        border: none;
        padding: 1rem 3rem;
        border-radius: 12px;
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(139, 92, 246, 0.6);
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);
    }
    
    /* File Uploader - Dark Theme */
    [data-testid="stFileUploader"] {
        background: rgba(30, 15, 50, 0.4);
        border: 2px dashed rgba(139, 92, 246, 0.5);
        border-radius: 16px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"] label {
        color: #c4b5fd !important;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Hide file upload limit text */
    [data-testid="stFileUploader"] small {
        display: none !important;
    }
    
    /* Dataframe Styling */
    [data-testid="stDataFrame"] {
        background: rgba(30, 15, 50, 0.6);
        border-radius: 12px;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }
    
    /* Info/Success/Warning Boxes */
    .stAlert {
        background: rgba(30, 15, 50, 0.7);
        border-left: 4px solid #8b5cf6;
        border-radius: 8px;
        color: #e0e7ff;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #8b5cf6 0%, #7c3aed 100%);
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.6);
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: rgba(139, 92, 246, 0.3);
    }
    
    .stSlider [role="slider"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.8);
    }
    
    /* Multiselect */
    .stMultiSelect {
        background: rgba(30, 15, 50, 0.4);
        border-radius: 8px;
    }
    
    /* Text Color */
    p, span, label {
        color: #e0e7ff;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
    }
    
    /* Welcome Card */
    .welcome-card {
        background: linear-gradient(135deg, rgba(30, 15, 50, 0.8) 0%, rgba(20, 10, 40, 0.9) 100%);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 20px;
        padding: 3rem;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    }
    
    /* Chart Container */
    .chart-container {
        background: rgba(30, 15, 50, 0.5);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(139, 92, 246, 0.2);
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 3rem 0;
        margin-top: 4rem;
        border-top: 1px solid rgba(139, 92, 246, 0.2);
        color: #a78bfa;
    }
    
    /* Download Button Special Styling */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 10px;
    }
    
    .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
    }
    
    /* Processing Status Card */
    .processing-status {
        background: rgba(30, 15, 50, 0.8);
        border: 1px solid rgba(139, 92, 246, 0.4);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Unlimited Badge */
    .unlimited-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        margin: 0.5rem 0.25rem;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }

    /* Integrated Cleaner Styles */
    .card {
        background: rgba(30, 15, 50, 0.6);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .card-green  { border-left: 4px solid #10b981; }
    .card-red    { border-left: 4px solid #ef4444; }
    .card-blue   { border-left: 4px solid #3b82f6; }
    .card-purple { border-left: 4px solid #8b5cf6; }
</style>
""", unsafe_allow_html=True)

# ===================== UTILITY FUNCTIONS =====================

def assess_image_quality(image_path):
    """Comprehensive image quality assessment"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0, {"error": "Cannot read image"}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_score = min(100, (laplacian.var() / 500) * 100)
        
        # Brightness
        brightness = np.mean(gray)
        brightness_score = 100 - (abs(127 - brightness) / 127 * 100)
        
        # Contrast
        contrast = np.std(gray)
        contrast_score = min(100, (contrast / 64) * 100)
        
        # Blur detection
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        blur_score = min(100, np.mean(magnitude_spectrum) / 2)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_score = min(100, (np.sum(edges > 0) / edges.size) * 500)
        
        # Combined score
        quality_score = (
            sharpness_score * 0.35 +
            brightness_score * 0.15 +
            contrast_score * 0.20 +
            blur_score * 0.15 +
            edge_score * 0.15
        )
        
        return quality_score, {
            "overall": round(quality_score, 2),
            "sharpness": round(sharpness_score, 2),
            "brightness": round(brightness_score, 2),
            "contrast": round(contrast_score, 2),
        }
    except Exception as e:
        return 0, {"error": f"Processing failed: {str(e)}"}


def resize_image(image_path, target_size, output_path):
    """High-quality image resizing - Returns the resized PIL image for further processing"""
    try:
        img = Image.open(image_path).convert('RGB')
        orig_w, orig_h = img.size
        target_w, target_h = target_size
        
        ratio = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
        
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        final_img = Image.new('RGB', target_size, (255, 255, 255))
        
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        final_img.paste(img_resized, (paste_x, paste_y))
        
        # Save the resized image
        final_img.save(output_path, 'JPEG', quality=95)
        return final_img  # Return PIL image for cleaning
    except Exception as e:
        st.error(f"Resize error: {e}")
        return None


def build_removal_mask(img_bgr, category_clean, ocr_reader, extra_margin):
    """Build removal mask for cleaning (from integrated Code 2)"""
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    cfg = SETTINGS[category_clean]
    keywords = KEYWORDS[category_clean]
    ocr_margin = cfg["ocr_margin"] + extra_margin * 0.10

    # METHOD 1: OCR
    if ocr_reader is not None:
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            results = ocr_reader.readtext(gray)
            for det in results:
                bbox, text = det[0], det[1]
                pts = np.array(bbox, dtype=np.int32)
                cx = np.mean(pts[:, 0])
                cy = np.mean(pts[:, 1])
                at_edge = (cx < w * ocr_margin or cx > w * (1 - ocr_margin) or
                           cy < h * ocr_margin or cy > h * (1 - ocr_margin))
                is_keyword = any(k in text.lower() for k in keywords)
                if at_edge or is_keyword:
                    exp = pts.copy()
                    exp[:, 0] = np.clip(pts[:, 0] + np.array([-10, 10, 10, -10]), 0, w)
                    exp[:, 1] = np.clip(pts[:, 1] + np.array([-10, -10, 10, 10]), 0, h)
                    cv2.fillPoly(mask, [exp.astype(np.int32)], 255)
        except Exception as e:
            pass  # Silent fail for OCR

    # METHOD 2: Corner & edge blanking
    cx_pct = cfg["corner_pct"] + extra_margin * 0.05
    ey_pct = cfg["edge_pct"] + extra_margin * 0.04
    cxs = int(w * cx_pct)
    cys = int(h * cx_pct)
    es = int(min(h, w) * ey_pct)

    mask[0:cys, 0:cxs] = 255
    mask[0:cys, w-cxs:w] = 255
    mask[h-cys:h, 0:cxs] = 255
    mask[h-cys:h, w-cxs:w] = 255
    mask[0:es, :] = 255
    mask[h-es:h, :] = 255
    mask[:, 0:es] = 255
    mask[:, w-es:w] = 255

    # METHOD 3: Isolated contour detection
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = h * w
    edge_check = 0.22

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, cw, ch = cv2.boundingRect(cnt)
        near_edge = (x < w * edge_check or x + cw > w * (1 - edge_check) or
                     y < h * edge_check or y + ch > h * (1 - edge_check))
        is_small = area < total_area * 0.04
        aspect = max(cw, ch) / (min(cw, ch) + 1)
        is_text_shape = aspect > 3.5
        if near_edge and (is_small or is_text_shape):
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            xp = max(5, int(cw * 0.1))
            yp = max(5, int(ch * 0.1))
            cv2.rectangle(mask,
                          (max(0, x - xp), max(0, y - yp)),
                          (min(w, x + cw + xp), min(h, y + ch + yp)), 255, -1)

    # METHOD 4: Legend box detection (plan images only)
    if category_clean in ("Master Plan", "Floor Plan"):
        edges_det = cv2.Canny(gray, 50, 150)
        kr = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        closed = cv2.morphologyEx(edges_det, cv2.MORPH_CLOSE, kr)
        rc, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in rc:
            x, y, cw, ch = cv2.boundingRect(cnt)
            a = cw * ch
            is_legend = 0.01 < (a / total_area) < 0.15
            near = (x < w * 0.25 or x + cw > w * 0.75 or
                    y < h * 0.25 or y + ch > h * 0.75)
            asp = max(cw, ch) / (min(cw, ch) + 1)
            if is_legend and near and 1 < asp < 4:
                cv2.rectangle(mask, (x - 10, y - 10),
                              (x + cw + 10, y + ch + 10), 255, -1)

    # Finalize
    k = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, k, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def find_main_plan_area(img_bgr):
    """Find main plan area to protect (from integrated Code 2)"""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            if cv2.contourArea(cnt) > h * w * 0.15:
                cv2.drawContours(main_mask, [cnt], -1, 255, -1)
    else:
        cv2.rectangle(main_mask, (int(w * 0.2), int(h * 0.2)),
                      (int(w * 0.8), int(h * 0.8)), 255, -1)
    kp = np.ones((10, 10), np.uint8)
    main_mask = cv2.dilate(main_mask, kp, iterations=1)
    return main_mask


def clean_image(pil_image, category_raw, ocr_reader, protect_center, extra_margin, inpaint_radius):
    """Clean image by removing text/logos/watermarks (integrated from Code 2)"""
    if category_raw == "rejected":
        return pil_image, 0.0   # ❌ DO NOT CLEAN REJECTED

    category_clean_map = {
        "masterplan": "Master Plan",
        "floorplan": "Floor Plan",
        "gallery": "Gallery"
    }

    if category_raw not in category_clean_map:
        return pil_image, 0.0   # ❌ UNKNOWN → SKIP CLEANING

    category_clean = category_clean_map[category_raw]
    try:
        img_arr = np.array(pil_image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)

        mask = build_removal_mask(img_bgr, category_clean, ocr_reader, extra_margin)

        if protect_center:
            main_area = find_main_plan_area(img_bgr)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(main_area))

        result = cv2.inpaint(img_bgr, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
        removed_pct = round(np.sum(mask == 255) / mask.size * 100, 2)
        result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        return result_pil, removed_pct
    except Exception as e:
        st.error(f"Cleaning error: {e}")
        return pil_image, 0.0  # Return original if failed


def process_single_image(image_path, filename, conf_thresh, qual_thresh, ocr_reader, protect_center, extra_margin, inpaint_radius, file_size=None):
    """Process single image - INTEGRATED WITH CLEANING STEP"""
    try:
        # STEP 1: AI Classification with predict_image
        raw_label, conf = predict_image(image_path)
        
        # STEP 2: Normalize the label using ENHANCED function
        normalized_label = normalize_category(raw_label)
        
        # STEP 3: Apply confidence threshold
        category = normalized_label if conf >= conf_thresh else "rejected"
        
        # STEP 4: Quality assessment
        quality_score, metrics = assess_image_quality(image_path)
        
        quality_status = "Good Quality" if quality_score >= qual_thresh else "Bad Quality"
        
        # Get original dimensions
        with Image.open(image_path) as img:
            width, height = img.size
        
        # STEP 5: Create output directory
        quality_folder = "good_quality" if quality_score >= qual_thresh else "bad_quality"
        output_dir = os.path.join(OUTPUT_DIR, category, quality_folder)
        os.makedirs(output_dir, exist_ok=True)
        
        base, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{base}.jpg")
        
        counter = 1
        while os.path.exists(output_path):
            output_path = os.path.join(output_dir, f"{base}_{counter}.jpg")
            counter += 1
        
        resize_info = "Original"
        cleaned_image = None
        removed_pct = 0.0
        
        # STEP 6: Resize (if not rejected and has target size)
        if category != "rejected" and CATEGORY_SIZES[category]:
            resized_pil = resize_image(image_path, CATEGORY_SIZES[category], output_path)
            if resized_pil:
                resize_info = f"{CATEGORY_SIZES[category][0]}×{CATEGORY_SIZES[category][1]}"
                # STEP 7: Clean the resized image
                cleaned_image, removed_pct = clean_image(
                    resized_pil, category, ocr_reader, protect_center, extra_margin, inpaint_radius
                )
                # STEP 8: Save cleaned image
                cleaned_image.save(output_path, 'JPEG', quality=95)
        else:
            # For rejected or no target size, just copy
            shutil.copy2(image_path, output_path)
            cleaned_image = Image.open(output_path).convert('RGB')
        
        # STEP 9: Return comprehensive result
        return {
            "filename": filename,
            "file_size": format_file_size(file_size) if file_size else "N/A",
            "category": CATEGORY_MAPPING.get(category, category),
            "category_raw": category,
            "model_label": raw_label,
            "confidence": conf * 100,
            "quality_status": quality_status,
            "quality_score": quality_score,
            "sharpness": metrics.get("sharpness", 0),
            "brightness": metrics.get("brightness", 0),
            "contrast": metrics.get("contrast", 0),
            "resolution": f"{width}×{height}",
            "output_size": resize_info,
            "cleaned_area_pct": removed_pct,
            "status": "success"
        }
    except Exception as e:
        return {
            "filename": filename,
            "file_size": format_file_size(file_size) if file_size else "N/A",
            "category": "Error",
            "category_raw": "error",
            "model_label": "N/A",
            "confidence": 0,
            "quality_status": "Error",
            "quality_score": 0,
            "sharpness": 0,
            "brightness": 0,
            "contrast": 0,
            "resolution": "N/A",
            "output_size": "N/A",
            "cleaned_area_pct": 0.0,
            "status": "failed",
            "error": str(e)
        }

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h2 style='color: #8b5cf6; margin: 0;'>⚙️ Configuration</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    confidence_threshold = st.slider(
        "🎯 Confidence Threshold",
        0, 100, 70, 5,
        help="Minimum confidence to accept classification"
    )
    
    quality_threshold = st.slider(
        "🌟 Quality Threshold",
        0, 100, 50, 5,
        help="Minimum quality score for 'Good Quality'"
    )
    
    st.markdown("---")
    
    # NEW: Cleaning Settings Section
    st.markdown("<h3 style='color: #a78bfa;'>🧹 AI Image Cleaning</h3>", unsafe_allow_html=True)
    
    protect_center = st.checkbox("🛡️ Protect Central Area", value=True,
                                 help="Prevents removal of main plan content in center")
    
    extra_margin = st.slider("🔍 Cleaning Aggressiveness", 0, 5, 2,
                             help="Higher = removes more peripheral content (logos/text)")
    
    inpaint_radius = st.slider("🎨 Inpaint Smoothness", 1, 8, 3,
                               help="Radius used to fill removed regions smoothly")
    
    st.markdown("""
    <div class='card card-green'>
        <p style='color:#10b981; font-weight:700; margin:0 0 6px;'>✅ What's Preserved</p>
        <p style='color:#c4b5fd; font-size:0.82rem; margin:0;'>
        • Building layouts & plans<br>
        • Unit / room labels inside<br>
        • Parking & amenity labels<br>
        • Gallery photo content<br>
        • All architectural drawings
        </p>
    </div>
    <div class='card card-red'>
        <p style='color:#ef4444; font-weight:700; margin:0 0 6px;'>🗑️ What Gets Removed</p>
        <p style='color:#c4b5fd; font-size:0.82rem; margin:0;'>
        • Company logos (all corners)<br>
        • MASTER / FLOOR PLAN titles<br>
        • Website URLs & watermarks<br>
        • Legend & scale boxes<br>
        • North arrows & direction text<br>
        • Copyright / disclaimer text
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Unlimited processing info
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.15) 100%); 
         padding: 1.5rem; border-radius: 12px; border: 2px solid rgba(16, 185, 129, 0.5);'>
        <h3 style='color: #10b981; margin-top: 0; text-align: center;'>∞ UNLIMITED ∞</h3>
        <p style='color: #86efac; font-weight: 600; margin: 0.5rem 0; text-align: center;'>✓ Unlimited Images</p>
        <p style='color: #86efac; font-weight: 600; margin: 0.5rem 0; text-align: center;'>✓ Unlimited File Size</p>
        <p style='color: #e0e7ff; margin: 0.5rem 0; font-size: 0.85rem; text-align: center;'>No restrictions!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(139, 92, 246, 0.3);'>
        <h3 style='color: #a78bfa; margin-top: 0;'>📋 Pipeline</h3>
        <p style='color: #e0e7ff; margin: 0.5rem 0;'>✓ Upload Images</p>
        <p style='color: #e0e7ff; margin: 0.5rem 0;'>✓ AI Classification</p>
        <p style='color: #e0e7ff; margin: 0.5rem 0;'>✓ Quality Analysis</p>
        <p style='color: #e0e7ff; margin: 0.5rem 0;'>✓ Auto-Resize</p>
        <p style='color: #10b981; font-weight: 700; margin: 0.5rem 0;'>🆕 AI Cleaning (Logos/Text)</p>
        <p style='color: #e0e7ff; margin: 0.5rem 0;'>✓ Organize & Save</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(139, 92, 246, 0.3);'>
        <h3 style='color: #a78bfa; margin-top: 0;'>📐 Output Sizes</h3>
        <p style='color: #e0e7ff; margin: 0.5rem 0;'>🏠 Floor Plans: 1500×1500</p>
        <p style='color: #e0e7ff; margin: 0.5rem 0;'>🗺️ Master Plans: 1640×860</p>
        <p style='color: #e0e7ff; margin: 0.5rem 0;'>🖼️ Gallery: 820×430</p>
        <p style='color: #e0e7ff; margin: 0.5rem 0;'>❌ Rejected: Original</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Reset statistics button
    if st.button("🔄 Reset All Statistics", use_container_width=True):
        if os.path.exists(STATS_FILE):
            os.remove(STATS_FILE)
            st.success("Statistics reset successfully!")
            st.rerun()

# Load OCR globally
ocr_reader = load_ocr()
if ocr_reader:
    st.sidebar.success("✅ AI OCR Engine Ready", icon="🤖")
else:
    st.sidebar.warning("⚠️ OCR unavailable — edge-based cleaning only", icon="⚠️")

# ===================== MAIN HEADER =====================
st.markdown("""
<div class="main-header">
    <div class="header-content">
        <div class="header-text">
            <h1>🏠 HOMES247</h1>
            <p>India's Favourite Property Portal - AI Dashboard</p>
            <div style='margin-top: 1rem;'>
                <span class='unlimited-badge'>∞ UNLIMITED IMAGES</span>
                <span class='unlimited-badge'>∞ UNLIMITED SIZE</span>
                <span class='unlimited-badge'>🆕 AI CLEANING</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===================== PERSISTENT STATISTICS BANNER =====================
stats = load_statistics()

if stats['total_processed'] > 0:
    st.markdown(f"""
    <div class="stats-banner">
        <h3 style='color: #10b981; margin: 0 0 1rem 0; text-align: center; font-size: 1.5rem;'>
            📊 ALL-TIME PROCESSING STATISTICS
        </h3>
        <div class="stats-grid">
            <div class="stat-item">
                <p class="stat-label">Total Processed</p>
                <p class="stat-value">{stats['total_processed']:,}</p>
            </div>
            <div class="stat-item">
                <p class="stat-label">🏠 Floor Plans</p>
                <p class="stat-value">{stats['floorplan_count']:,}</p>
            </div>
            <div class="stat-item">
                <p class="stat-label">🗺️ Master Plans</p>
                <p class="stat-value">{stats['masterplan_count']:,}</p>
            </div>
            <div class="stat-item">
                <p class="stat-label">🖼️ Gallery</p>
                <p class="stat-value">{stats['gallery_count']:,}</p>
            </div>
            <div class="stat-item">
                <p class="stat-label">❌ Rejected</p>
                <p class="stat-value">{stats['rejected_count']:,}</p>
            </div>
            <div class="stat-item">
                <p class="stat-label">✅ Good Quality</p>
                <p class="stat-value">{stats['good_quality_count']:,}</p>
            </div>
            <div class="stat-item">
                <p class="stat-label">⚠️ Bad Quality</p>
                <p class="stat-value">{stats['bad_quality_count']:,}</p>
            </div>
            <div class="stat-item">
                <p class="stat-label">🔢 Sessions</p>
                <p class="stat-value">{stats['total_sessions']:,}</p>
            </div>
        </div>
        <div style='margin-top: 1.5rem; text-align: center; padding-top: 1rem; border-top: 1px solid rgba(139, 92, 246, 0.3);'>
            <p style='color: #c4b5fd; margin: 0; font-size: 0.9rem;'>
                📅 First Upload: <strong style='color: #10b981;'>{stats['first_upload_date'] if stats['first_upload_date'] else 'N/A'}</strong>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                🕐 Last Upload: <strong style='color: #10b981;'>{stats['last_upload_date'] if stats['last_upload_date'] else 'N/A'}</strong>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ===================== SESSION STATE =====================
if 'results' not in st.session_state:
    st.session_state.results = []
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'processing' not in st.session_state:
    st.session_state.processing = False

# ===================== UPLOAD SECTION =====================
st.markdown("""
<div class="section-header">
    <h2>📤 Upload Real Estate Images - COMPLETELY UNLIMITED</h2>
</div>
""", unsafe_allow_html=True)

# Unlimited info banner
st.markdown("""
<div style='background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.1) 100%); 
     border: 2px solid #10b981; border-radius: 12px; padding: 1.5rem; margin: 1rem 0;'>
    <h3 style='color: #10b981; margin: 0; text-align: center; font-size: 1.5rem;'>
        ⚡ NO LIMITS - PROCESS ANY NUMBER OF IMAGES OF ANY SIZE ⚡
    </h3>
    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;'>
        <div style='text-align: center; background: rgba(30, 15, 50, 0.5); padding: 1rem; border-radius: 8px;'>
            <p style='color: #86efac; font-size: 2rem; font-weight: 700; margin: 0;'>∞</p>
            <p style='color: #e0e7ff; font-size: 1rem; margin: 0.5rem 0; font-weight: 600;'>Unlimited Quantity</p>
            <p style='color: #a78bfa; font-size: 0.85rem; margin: 0;'>Upload 1000+ images!</p>
        </div>
        <div style='text-align: center; background: rgba(30, 15, 50, 0.5); padding: 1rem; border-radius: 8px;'>
            <p style='color: #86efac; font-size: 2rem; font-weight: 700; margin: 0;'>∞</p>
            <p style='color: #e0e7ff; font-size: 1rem; margin: 0.5rem 0; font-weight: 600;'>Unlimited File Size</p>
            <p style='color: #a78bfa; font-size: 0.85rem; margin: 0;'>From KB to GB+!</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Drag & Drop or Browse - Upload ANY number of images with ANY file size",
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=True,
    help="UNLIMITED: Upload as many images as you need with any file size"
)

if uploaded_files:
    total_images = len(uploaded_files)
    total_size = sum(file.size for file in uploaded_files)
    
    # Display upload statistics
    st.markdown(f"""
    <div style='background: rgba(139, 92, 246, 0.15); border-radius: 12px; padding: 1.5rem; margin: 1rem 0; border: 1px solid rgba(139, 92, 246, 0.4);'>
        <h3 style='color: #8b5cf6; margin: 0;'>📊 Upload Statistics</h3>
        <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;'>
            <div style='text-align: center; background: rgba(30, 15, 50, 0.4); padding: 1rem; border-radius: 8px;'>
                <p style='color: #c4b5fd; font-size: 0.9rem; margin: 0;'>Total Images</p>
                <p style='color: #ffffff; font-size: 2rem; font-weight: 700; margin: 0.5rem 0;'>{total_images}</p>
            </div>
            <div style='text-align: center; background: rgba(30, 15, 50, 0.4); padding: 1rem; border-radius: 8px;'>
                <p style='color: #c4b5fd; font-size: 0.9rem; margin: 0;'>Total Size</p>
                <p style='color: #ffffff; font-size: 2rem; font-weight: 700; margin: 0.5rem 0;'>{format_file_size(total_size)}</p>
            </div>
            <div style='text-align: center; background: rgba(30, 15, 50, 0.4); padding: 1rem; border-radius: 8px;'>
                <p style='color: #c4b5fd; font-size: 0.9rem; margin: 0;'>Average Size</p>
                <p style='color: #ffffff; font-size: 2rem; font-weight: 700; margin: 0.5rem 0;'>{format_file_size(total_size/total_images)}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.success(f"✅ **{total_images} images** ({format_file_size(total_size)}) ready for processing!")
        
        # Show estimate
        estimated_time = total_images * 3  # Slightly longer due to cleaning
        st.info(f"⏱️ Estimated processing time: ~{estimated_time // 60} min {estimated_time % 60} sec (incl. AI Cleaning)")
        
        if st.button("🚀 PROCESS ALL IMAGES", use_container_width=True, disabled=st.session_state.processing):
            st.session_state.results = []
            st.session_state.processed = False
            st.session_state.processing = True
            
            # Create progress tracking
            progress_bar = st.progress(0)
            status_container = st.container()
            
            # Processing metrics
            start_time = time.time()
            success_count = 0
            failed_count = 0
            total_processed_size = 0
            
            # Status display
            with status_container:
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                metric_total = col_stat1.empty()
                metric_success = col_stat2.empty()
                metric_failed = col_stat3.empty()
                metric_size = col_stat4.empty()
                status_text = st.empty()
            
            # Process all images without limit
            for idx, file in enumerate(uploaded_files):
                current_num = idx + 1
                file_size = file.size
                total_processed_size += file_size
                
                # Update status
                status_text.markdown(f"""
                <div class="processing-status">
                    <h4 style='color: #8b5cf6; margin: 0;'>⏳ Processing Image {current_num}/{total_images}</h4>
                    <p style='color: #e0e7ff; margin: 0.5rem 0;'><strong>File:</strong> {file.name}</p>
                    <p style='color: #a78bfa; margin: 0.3rem 0; font-size: 0.9rem;'><strong>Size:</strong> {format_file_size(file_size)}</p>
                    <p style='color: #a78bfa; margin: 0; font-size: 0.9rem;'>Progress: {(current_num/total_images)*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Update metrics
                metric_total.metric("📊 Processed", f"{current_num}/{total_images}")
                metric_success.metric("✅ Success", success_count)
                metric_failed.metric("❌ Failed", failed_count)
                metric_size.metric("💾 Processed", format_file_size(total_processed_size))
                
                try:
                    # Save uploaded file
                    temp_path = os.path.join(UPLOAD_DIR, file.name)
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Process image with integrated cleaning
                    result = process_single_image(
                        temp_path, file.name,
                        confidence_threshold / 100,
                        quality_threshold,
                        ocr_reader, protect_center, extra_margin, inpaint_radius,
                        file_size
                    )
                    
                    st.session_state.results.append(result)
                    
                    if result['status'] == 'success':
                        success_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    failed_count += 1
                    st.session_state.results.append({
                        "filename": file.name,
                        "file_size": format_file_size(file_size),
                        "category": "Error",
                        "category_raw": "error",
                        "model_label": "N/A",
                        "confidence": 0,
                        "quality_status": "Error",
                        "quality_score": 0,
                        "sharpness": 0,
                        "brightness": 0,
                        "contrast": 0,
                        "resolution": "N/A",
                        "output_size": "N/A",
                        "cleaned_area_pct": 0.0,
                        "status": "failed",
                        "error": str(e)
                    })
                
                # Update progress bar
                progress_bar.progress(current_num / total_images)
            
            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Update persistent statistics
            updated_stats = update_statistics(st.session_state.results)
            
            # Final status
            st.session_state.processed = True
            st.session_state.processing = False
            status_text.empty()
            progress_bar.empty()
            
            # Show completion message
            st.balloons()
            st.success(f"""
            🎉 **Processing Complete!**
            
            - Total Images: **{total_images}**
            - Total Data: **{format_file_size(total_size)}**
            - Successfully Processed: **{success_count}**
            - Failed: **{failed_count}**
            - Time Taken: **{processing_time//60:.0f} min {processing_time%60:.0f} sec**
            - Average Time/Image: **{processing_time/total_images:.2f} sec**
            - Processing Speed: **{format_file_size(total_size/processing_time)}/sec**
            
            📊 **Statistics Updated!** Refresh the page to see updated all-time counts.
            🆕 **AI Cleaning Applied:** Logos, text, and watermarks removed where detected.
            """)
            
            # Auto-refresh to show updated statistics
            time.sleep(2)
            st.rerun()

# ===================== RESULTS DISPLAY =====================
if st.session_state.processed and st.session_state.results:
    df = pd.DataFrame(st.session_state.results)
    
    # METRICS
    st.markdown("""
    <div class="section-header">
        <h2>📊 Current Session Analytics</h2>
    </div>
    """, unsafe_allow_html=True)
    
    total = len(df)
    successful = len(df[df['status'] == 'success'])
    property_img = len(df[df['category_raw'] == 'gallery'])
    floor_img = len(df[df['category_raw'] == 'floorplan'])
    master_img = len(df[df['category_raw'] == 'masterplan'])
    rejected_img = len(df[df['category_raw'] == 'rejected'])
    good_qual = len(df[df['quality_status'] == 'Good Quality'])
    bad_qual = len(df[df['quality_status'] == 'Bad Quality'])
    
    # Calculate averages only for successful images
    df_success = df[df['status'] == 'success']
    avg_conf = df_success['confidence'].mean() if len(df_success) > 0 else 0
    avg_qual = df_success['quality_score'].mean() if len(df_success) > 0 else 0
    avg_cleaned = df_success['cleaned_area_pct'].mean() if len(df_success) > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📁 Total Images", total)
        st.metric("✅ Successful", successful, f"{successful/total*100:.1f}%")
    
    with col2:
        st.metric("🖼️ Gallery Images", property_img)
        st.metric("📐 Floor Plans", floor_img)
    
    with col3:
        st.metric("🗺️ Master Plans", master_img)
        st.metric("❌ Rejected", rejected_img)
    
    with col4:
        st.metric("🌟 Good Quality", good_qual, f"{good_qual/total*100:.1f}%")
        st.metric("⚠️ Bad Quality", bad_qual, f"{bad_qual/total*100:.1f}%")
    
    # Additional metrics row
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("🎯 Avg Confidence", f"{avg_conf:.1f}%")
    
    with col6:
        st.metric("💎 Avg Quality", f"{avg_qual:.1f}%")
    
    with col7:
        failed = total - successful
        st.metric("⚠️ Failed", failed)
    
    with col8:
        st.metric("🧹 Avg Cleaned Area", f"{avg_cleaned:.1f}%")
    
    # VISUALIZATIONS
    st.markdown("""
    <div class="section-header">
        <h2>📈 Visual Analytics</h2>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Dark theme template
        template = {
            'layout': {
                'paper_bgcolor': 'rgba(30, 15, 50, 0.5)',
                'plot_bgcolor': 'rgba(20, 10, 40, 0.3)',
                'font': {'color': '#e0e7ff'},
                'xaxis': {'gridcolor': 'rgba(139, 92, 246, 0.1)'},
                'yaxis': {'gridcolor': 'rgba(139, 92, 246, 0.1)'}
            }
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            cat_counts = df['category'].value_counts().reset_index()
            cat_counts.columns = ['Category', 'Count']
            
            fig1 = px.bar(
                cat_counts, x='Category', y='Count',
                title='📊 Category Distribution',
                color='Category',
                color_discrete_sequence=['#8b5cf6', '#7c3aed', '#6d28d9', '#5b21b6']
            )
            fig1.update_layout(template['layout'], height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            qual_counts = df['quality_status'].value_counts().reset_index()
            qual_counts.columns = ['Quality', 'Count']
            
            fig2 = px.pie(
                qual_counts, names='Quality', values='Count',
                title='🎯 Quality Distribution',
                color_discrete_sequence=['#10b981', '#ef4444', '#6b7280']
            )
            fig2.update_layout(template['layout'], height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            if len(df_success) > 0:
                qual_by_cat = df_success.groupby('category')['quality_score'].mean().reset_index()
                
                fig3 = px.bar(
                    qual_by_cat, x='category', y='quality_score',
                    title='📈 Average Quality by Category',
                    color='quality_score',
                    color_continuous_scale='Viridis'
                )
                fig3.update_layout(template['layout'], height=400)
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.warning("No successful images to display quality by category")
        
        with col4:
            if len(df_success) > 0:
                fig4 = px.histogram(
                    df_success, x='confidence', nbins=20,
                    title='📊 Confidence Distribution',
                    color_discrete_sequence=['#8b5cf6']
                )
                fig4.update_layout(template['layout'], height=400)
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.warning("No successful images to display confidence distribution")
        
        # NEW: Cleaning visualization
        col5, col6 = st.columns(2)
        
        with col5:
            if len(df_success) > 0:
                cleaned_by_cat = df_success.groupby('category')['cleaned_area_pct'].mean().reset_index()
                
                fig5 = px.bar(
                    cleaned_by_cat, x='category', y='cleaned_area_pct',
                    title='🧹 Average Cleaned Area by Category (%)',
                    color='cleaned_area_pct',
                    color_continuous_scale='Reds'
                )
                fig5.update_layout(template['layout'], height=400)
                st.plotly_chart(fig5, use_container_width=True)
            else:
                st.warning("No successful images to display cleaning metrics")
        
        with col6:
            if len(df_success) > 0:
                fig6 = px.histogram(
                    df_success, x='cleaned_area_pct', nbins=20,
                    title='📊 Cleaning Area Distribution (%)',
                    color_discrete_sequence=['#ef4444']
                )
                fig6.update_layout(template['layout'], height=400)
                st.plotly_chart(fig6, use_container_width=True)
            else:
                st.warning("No successful images to display cleaning distribution")
            
    except ImportError:
        st.warning("⚠️ Install plotly: `pip install plotly`")
    
    # DATA TABLE
    st.markdown("""
    <div class="section-header">
        <h2>📋 Detailed Results Table</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cat_filter = st.multiselect(
            "🏷️ Filter by Category",
            options=df['category'].unique().tolist(),
            default=df['category'].unique().tolist()
        )
    
    with col2:
        qual_filter = st.multiselect(
            "🌟 Filter by Quality",
            options=df['quality_status'].unique().tolist(),
            default=df['quality_status'].unique().tolist()
        )
    
    with col3:
        status_filter = st.multiselect(
            "📊 Filter by Status",
            options=df['status'].unique().tolist(),
            default=df['status'].unique().tolist()
        )
    
    filtered_df = df[
        (df['category'].isin(cat_filter)) &
        (df['quality_status'].isin(qual_filter)) &
        (df['status'].isin(status_filter))
    ]
    
    st.info(f"📋 Showing **{len(filtered_df)}** of **{total}** images")
    
    # Display table with new cleaning column
    display_columns = ['filename', 'file_size', 'category', 'model_label', 'confidence', 'quality_status', 
                       'quality_score', 'resolution', 'output_size', 'cleaned_area_pct', 'status']
    
    if 'error' in filtered_df.columns and filtered_df['error'].notna().any():
        display_columns.append('error')
    
    st.dataframe(
        filtered_df[display_columns],
        use_container_width=True,
        height=400
    )
    
    # DOWNLOADS
    st.markdown("""
    <div class="section-header">
        <h2>💾 Download Reports & Images</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with col1:
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📄 CSV Report",
            csv,
            f"homes247_{timestamp}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        json_data = filtered_df.to_json(orient='records', indent=2)
        st.download_button(
            "📋 JSON Report",
            json_data,
            f"homes247_{timestamp}.json",
            "application/json",
            use_container_width=True
        )
    
    with col3:
        try:
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='Results')
                
                summary_data = {
                    'Metric': ['Total Images', 'Successful', 'Failed', 'Gallery Images', 
                               'Floor Plans', 'Master Plans', 'Rejected', 
                               'Good Quality', 'Bad Quality', 'Avg Confidence', 'Avg Quality', 'Avg Cleaned Area (%)'],
                    'Value': [total, successful, total-successful, property_img, 
                             floor_img, master_img, rejected_img,
                             good_qual, bad_qual, f"{avg_conf:.2f}%", f"{avg_qual:.2f}%", f"{avg_cleaned:.2f}%"]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, index=False, sheet_name='Summary')
            
            st.download_button(
                "📊 Excel Report",
                output.getvalue(),
                f"homes247_{timestamp}.xlsx",
                "application/vnd.ms-excel",
                use_container_width=True
            )
        except:
            st.info("📦 Install openpyxl: `pip install openpyxl`")
    
    with col4:
        try:
            import zipfile
            from io import BytesIO
            
            with st.spinner("📦 Creating ZIP..."):
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    zip_file.writestr(f"reports/homes247_report_{timestamp}.csv", 
                                     filtered_df.to_csv(index=False))
                    
                    zip_file.writestr(f"reports/homes247_report_{timestamp}.json", 
                                     filtered_df.to_json(orient='records', indent=2))
                    
                    images_added = 0
                    for _, row in filtered_df.iterrows():
                        if row['status'] == 'success':
                            category = row['category_raw']
                            quality_folder = "good_quality" if row['quality_status'] == "Good Quality" else "bad_quality"
                            
                            filename = row['filename']
                            base, _ = os.path.splitext(filename)
                            
                            possible_paths = [
                                os.path.join(OUTPUT_DIR, category, quality_folder, f"{base}.jpg"),
                                os.path.join(OUTPUT_DIR, category, quality_folder, f"{base}_1.jpg"),
                                os.path.join(OUTPUT_DIR, category, quality_folder, filename)
                            ]
                            
                            for img_path in possible_paths:
                                if os.path.exists(img_path):
                                    zip_path = f"images/{category}/{quality_folder}/{os.path.basename(img_path)}"
                                    with open(img_path, 'rb') as img_file:
                                        zip_file.writestr(zip_path, img_file.read())
                                    images_added += 1
                                    break
                    
                    summary_text = f"""HOMES247 - UNLIMITED PROCESSING SUMMARY
=========================================

Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Report ID: {timestamp}

UNLIMITED CAPABILITIES
----------------------
✓ No limit on number of images
✓ No limit on file size per image
✓ Process thousands of images with GB+ sizes
✓ Integrated AI Cleaning for logos/text/watermarks

STATISTICS
----------
Total Images: {total}
Successfully Processed: {successful} ({successful/total*100:.1f}%)
Failed: {total-successful}

CATEGORIES
----------
Gallery: {property_img} | Floor Plans: {floor_img}
Master Plans: {master_img} | Rejected: {rejected_img}

QUALITY
-------
Good Quality: {good_qual} ({good_qual/total*100:.1f}%)
Bad Quality: {bad_qual} ({bad_qual/total*100:.1f}%)

CLEANING METRICS
----------------
Avg Cleaned Area: {avg_cleaned:.2f}%

CONFIGURATION
-------------
Confidence Threshold: {confidence_threshold}%
Quality Threshold: {quality_threshold}%
Cleaning Aggressiveness: {extra_margin}
Inpaint Radius: {inpaint_radius}
Protect Center: {'Yes' if protect_center else 'No'}

OUTPUT SIZES
------------
Floor Plans: 1500×1500
Master Plans: 1640×860
Gallery: 820×430
Rejected: Original

---
Homes247 - India's Favourite Property Portal
UNLIMITED Edition | Powered by AI + EasyOCR + OpenCV
"""
                    zip_file.writestr("README.txt", summary_text)
                
                st.download_button(
                    "📦 Complete ZIP",
                    zip_buffer.getvalue(),
                    f"homes247_complete_{timestamp}.zip",
                    "application/zip",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error: {str(e)}")

else:
    # WELCOME SCREEN
    st.markdown("""
    <div class="welcome-card">
        <h2 style='color: #8b5cf6; text-align: center; font-size: 2.5rem; margin-bottom: 2rem;'>
            👋 Welcome to Homes247 AI Dashboard
        </h2>
        
        <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem; margin: 2rem 0;'>
            <div style='text-align: center;'>
                <div style='font-size: 3rem; margin-bottom: 1rem;'>🤖</div>
                <h3 style='color: #c4b5fd;'>AI Classification</h3>
                <p style='color: #a78bfa;'>Automatic categorization</p>
            </div>
            <div style='text-align: center;'>
                <div style='font-size: 3rem; margin-bottom: 1rem;'>🔍</div>
                <h3 style='color: #c4b5fd;'>Quality Analysis</h3>
                <p style='color: #a78bfa;'>Advanced metrics</p>
            </div>
            <div style='text-align: center;'>
                <div style='font-size: 3rem; margin-bottom: 1rem;'>🧹</div>
                <h3 style='color: #c4b5fd;'>AI Cleaning</h3>
                <p style='color: #a78bfa;'>Remove logos & text</p>
            </div>
        </div>
        
        <div style='background: linear-gradient(135deg, rgba(16, 185, 129, 0.3) 0%, rgba(5, 150, 105, 0.2) 100%); 
             border: 3px solid #10b981; border-radius: 16px; padding: 2.5rem; margin: 2rem 0;'>
            <h3 style='color: #10b981; text-align: center; margin-top: 0; font-size: 2rem;'>
                ⚡ COMPLETELY UNLIMITED ⚡
            </h3>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 1.5rem;'>
                <div style='background: rgba(30, 15, 50, 0.5); padding: 1.5rem; border-radius: 12px;'>
                    <p style='color: #86efac; text-align: center; font-size: 2rem; font-weight: 700; margin: 0;'>∞</p>
                    <h4 style='color: #e0e7ff; text-align: center; margin: 1rem 0;'>Unlimited Images</h4>
                    <p style='color: #a78bfa; text-align: center; margin: 0;'>Upload 1000+ images!</p>
                </div>
                <div style='background: rgba(30, 15, 50, 0.5); padding: 1.5rem; border-radius: 12px;'>
                    <p style='color: #86efac; text-align: center; font-size: 2rem; font-weight: 700; margin: 0;'>∞</p>
                    <h4 style='color: #e0e7ff; text-align: center; margin: 1rem 0;'>Unlimited File Size</h4>
                    <p style='color: #a78bfa; text-align: center; margin: 0;'>From KB to GB+!</p>
                </div>
            </div>
        </div>
        
        <h3 style='color: #c4b5fd; margin-top: 3rem;'>🚀 Quick Start:</h3>
        <ol style='color: #e0e7ff; font-size: 1.1rem; line-height: 2rem;'>
            <li>Upload images above (unlimited!)</li>
            <li>Adjust thresholds & cleaning settings (optional)</li>
            <li>Click "Process All Images"</li>
            <li>Download reports</li>
        </ol>
        
        <h3 style='color: #c4b5fd; margin-top: 2rem;'>✨ Features:</h3>
        <ul style='color: #e0e7ff; font-size: 1.1rem; line-height: 2rem;'>
            <li>✓ Unlimited images & file sizes</li>
            <li>✓ Persistent statistics tracking</li>
            <li>✓ Real-time progress tracking</li>
            <li>✓ Quality analysis</li>
            <li>✓ Auto-resize & organize</li>
            <li>✓ <strong>🆕 AI Cleaning:</strong> Remove logos, text, watermarks</li>
            <li>✓ Export CSV, JSON, Excel, ZIP</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# FOOTER
st.markdown("""
<div class="footer">
    <h2 style='color: #8b5cf6; margin-bottom: 1rem;'>🏠 HOMES247</h2>
    <p style='font-size: 1.2rem; margin: 0.5rem 0;'>India's Favourite Property Portal</p>
    <p style='font-size: 1rem; color: #10b981; font-weight: 600; margin: 0.5rem 0;'>∞ UNLIMITED EDITION ∞</p>
    <p style='font-size: 0.9rem; color: #6b7280; margin-top: 1rem;'>
        Version 2.0 Professional - Zero Limits + Persistent Statistics + AI Cleaning | Powered by AI + EasyOCR + OpenCV
    </p>
    <p style='font-size: 0.9rem; color: #6b7280;'>© 2026 Homes247. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)