"""
Core Image Processing Module
Extracted from demo.py for reusable pipeline functions
Version 1.0
"""

import os
import cv2
import numpy as np
from PIL import Image
import json
from datetime import datetime
import time as _time

# ===================== CONFIGURATION =====================
BASE_DIR = r"C:\Users\Homes247\Desktop\Bulk_image"
OUTPUT_DIR = os.path.join(BASE_DIR, "api_output")
TEMP_DIR = os.path.join(BASE_DIR, "api_temp")
WATERMARK_LOGO_FILE = os.path.join(BASE_DIR, "watermark_logo.png")
SETTINGS_FILE = os.path.join(BASE_DIR, "app_settings.json")

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
    "Master Plan": {"corner_pct": 0.06, "edge_pct": 0.02, "ocr_margin": 0.10},
    "Floor Plan":  {"corner_pct": 0.02, "edge_pct": 0.00, "ocr_margin": 0.06},
    "Gallery":     {"corner_pct": 0.01, "edge_pct": 0.00, "ocr_margin": 0.08},
}

CATEGORY_TO_TEXT_REMOVAL = {
    "floorplan": "Floor Plan",
    "masterplan": "Master Plan",
    "gallery": "Gallery",
    "rejected": None
}

# ===================== HELPER FUNCTIONS =====================
def load_app_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"output_format": "WEBP"}

def get_format_extension(fmt):
    mapping = {
        "WEBP": ("webp", "WEBP", [cv2.IMWRITE_WEBP_QUALITY, 100]),
        "JPEG": ("jpg",  "JPEG", [cv2.IMWRITE_JPEG_QUALITY, 100]),
        "JPG":  ("jpg",  "JPEG", [cv2.IMWRITE_JPEG_QUALITY, 100]),
        "PNG":  ("png",  "PNG",  [cv2.IMWRITE_PNG_COMPRESSION, 0]),
        "AVIF": ("jpg",  "JPEG", [cv2.IMWRITE_JPEG_QUALITY, 100]),
    }
    return mapping.get(fmt, ("webp", "WEBP", [cv2.IMWRITE_WEBP_QUALITY, 100]))

def ensure_directories():
    """Create necessary directories"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    for cat in ["floorplan", "masterplan", "gallery", "rejected"]:
        for quality in ["good_quality", "bad_quality"]:
            os.makedirs(os.path.join(OUTPUT_DIR, cat, quality), exist_ok=True)

# ===================== TEXT REMOVAL FUNCTIONS =====================
def build_removal_mask(img_bgr, category, ocr_reader, extra_margin=2, protect_center=True):
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cfg = TEXT_REMOVAL_SETTINGS.get(category, TEXT_REMOVAL_SETTINGS["Gallery"])
    keywords = TEXT_REMOVAL_KEYWORDS.get(category, TEXT_REMOVAL_KEYWORDS["Gallery"])
    ocr_margin = cfg["ocr_margin"] + extra_margin * 0.08

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

                if category == "Gallery":
                    text_lower = text.lower()
                    has_keyword = any(kw in text_lower for kw in keywords)
                    should_remove = at_edge and has_keyword
                else:
                    should_remove = at_edge

                if should_remove:
                    exp = pts.copy()
                    exp[:, 0] = np.clip(pts[:, 0] + np.array([-8, 8, 8, -8]), 0, w)
                    exp[:, 1] = np.clip(pts[:, 1] + np.array([-8, -8, 8, 8]), 0, h)
                    cv2.fillPoly(mask, [exp.astype(np.int32)], 255)
        except Exception as e:
            pass

    cx_pct = cfg["corner_pct"] + extra_margin * 0.01
    ey_pct = cfg["edge_pct"]
    cxs = int(w * cx_pct)
    cys = int(h * cx_pct)
    es  = int(min(h, w) * ey_pct)

    if category != "Floor Plan":
        mask[0:cys, 0:cxs] = 255
        mask[0:cys, w-cxs:w] = 255
        mask[h-cys:h, 0:cxs] = 255
        mask[h-cys:h, w-cxs:w] = 255

    if category == "Master Plan":
        mask[0:es, :] = 255
        mask[h-es:h, :] = 255
        mask[:, 0:es] = 255
        mask[:, w-es:w] = 255

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = h * w
    edge_check = 0.18
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, cw, ch = cv2.boundingRect(cnt)
        near_edge = (x < w * edge_check or x + cw > w * (1 - edge_check) or
                    y < h * edge_check or y + ch > h * (1 - edge_check))
        is_small = area < total_area * 0.03
        aspect = max(cw, ch) / (min(cw, ch) + 1)
        is_text_shape = aspect > 3.0
        if near_edge and (is_small or is_text_shape):
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            xp = max(5, int(cw * 0.08))
            yp = max(5, int(ch * 0.08))
            cv2.rectangle(mask,
                         (max(0, x - xp), max(0, y - yp)),
                         (min(w, x + cw + xp), min(h, y + ch + yp)),
                         255, -1)

    if category in ("Master Plan", "Floor Plan"):
        edges_det = cv2.Canny(gray, 50, 150)
        kr = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
        closed = cv2.morphologyEx(edges_det, cv2.MORPH_CLOSE, kr)
        rc, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in rc:
            x, y, cw, ch = cv2.boundingRect(cnt)
            a = cw * ch
            is_legend = 0.003 < (a / total_area) < 0.12
            at_edge_area = (x < w * 0.20 or x + cw > w * 0.80 or
                           y < h * 0.20 or y + ch > h * 0.80)
            asp = max(cw, ch) / (min(cw, ch) + 1)
            if is_legend and at_edge_area and 1.0 < asp < 4.0:
                cv2.rectangle(mask, (x - 8, y - 8), (x + cw + 8, y + ch + 8), 255, -1)

    if category in ("Master Plan", "Floor Plan"):
        main_area = find_main_plan_area(img_bgr)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(main_area))

    k = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(mask, k, iterations=1)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def find_main_plan_area(img_bgr):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
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
                     (int(w * 0.85), int(h * 0.85)),
                     255, -1)
    kp = np.ones((15, 15), np.uint8)
    main_mask = cv2.dilate(main_mask, kp, iterations=2)
    return main_mask


def remove_text_and_logos(image_path, output_path, category, ocr_reader,
                          protect_center=True, extra_margin=2, inpaint_radius=3):
    try:
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return False, 0.0, "Failed to read image"
        original_quality = cv2.Laplacian(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        mask = build_removal_mask(img_bgr, category, ocr_reader, extra_margin, protect_center)
        removed_pct = round(np.sum(mask == 255) / mask.size * 100, 2)
        if removed_pct > 25:
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=2)
            removed_pct = round(np.sum(mask == 255) / mask.size * 100, 2)
        if np.sum(mask) > 0:
            safe_radius = min(inpaint_radius, 3)
            if category in ("Floor Plan",):
                result = cv2.inpaint(img_bgr, mask, inpaintRadius=1, flags=cv2.INPAINT_NS)
            else:
                result = cv2.inpaint(img_bgr, mask, inpaintRadius=safe_radius, flags=cv2.INPAINT_TELEA)
            result_quality = cv2.Laplacian(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if result_quality < original_quality * 0.7:
                result = cv2.inpaint(img_bgr, mask, inpaintRadius=1, flags=cv2.INPAINT_NS)
        else:
            result = img_bgr.copy()

        _out_ext = os.path.splitext(output_path)[1].lower()
        if _out_ext in (".jpg", ".jpeg"):
            _cv_params = [cv2.IMWRITE_JPEG_QUALITY, 100]
        elif _out_ext == ".png":
            _cv_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
        elif _out_ext == ".webp":
            _cv_params = [cv2.IMWRITE_WEBP_QUALITY, 100]
        else:
            _cv_params = []

        cv2.imwrite(output_path, result, _cv_params)
        return True, removed_pct, None
    except Exception as e:
        return False, 0.0, str(e)


# ===================== WATERMARK FUNCTIONS =====================
def load_watermark_logo():
    if os.path.exists(WATERMARK_LOGO_FILE):
        try:
            logo = Image.open(WATERMARK_LOGO_FILE).convert("RGBA")
            return logo
        except Exception as e:
            print(f"Error loading watermark: {e}")
            return None
    return None


def apply_watermark_to_image(image_path, output_path, watermark_logo=None,
                             logo_size_ratio=0.05, logo_opacity=0.20):
    try:
        main_image = Image.open(image_path)
        if main_image.mode not in ('RGB', 'RGBA'):
            main_image = main_image.convert('RGB')
        img_width, img_height = main_image.size
        if watermark_logo is None:
            watermark_logo = load_watermark_logo()
        if watermark_logo is None:
            if main_image.mode == 'RGBA':
                main_image = main_image.convert('RGB')
            _fmt = load_app_settings().get("output_format", "WEBP")
            _ext, _pil_fmt, _ = get_format_extension(_fmt)
            _save_kwargs = {"quality": 100, "subsampling": 0} if _pil_fmt == "JPEG" else {"quality": 100} if _pil_fmt in ("WEBP", "AVIF") else {}
            main_image.save(output_path, _pil_fmt, **_save_kwargs)
            return True

        logo_width = int(img_width * logo_size_ratio)
        logo_width = max(logo_width, 40)
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
        if main_image.mode != 'RGBA':
            watermarked = main_image.convert('RGBA')
        else:
            watermarked = main_image.copy()
        watermarked.paste(logo_resized, (logo_x, logo_y), logo_resized)
        watermarked_rgb = watermarked.convert('RGB')
        _fmt = load_app_settings().get("output_format", "WEBP")
        _ext, _pil_fmt, _ = get_format_extension(_fmt)
        _save_kwargs = {"quality": 100, "subsampling": 0} if _pil_fmt == "JPEG" else {"quality": 100} if _pil_fmt in ("WEBP", "AVIF") else {}
        watermarked_rgb.save(output_path, _pil_fmt, **_save_kwargs)
        return True
    except Exception:
        try:
            img = Image.open(image_path)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            _fmt = load_app_settings().get("output_format", "WEBP")
            _ext, _pil_fmt, _ = get_format_extension(_fmt)
            _save_kwargs = {"quality": 100, "subsampling": 0} if _pil_fmt == "JPEG" else {"quality": 100} if _pil_fmt in ("WEBP", "AVIF") else {}
            img.save(output_path, _pil_fmt, **_save_kwargs)
        except:
            pass
        return False


# ===================== IMAGE RESIZING =====================
def resize_image(image_path, target_size, output_path):
    try:
        img = Image.open(image_path).convert('RGB')
        orig_w, orig_h = img.size
        target_w, target_h = target_size
        scale   = min(target_w / orig_w, target_h / orig_h)
        new_w   = int(orig_w * scale)
        new_h   = int(orig_h * scale)
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas  = Image.new('RGB', (target_w, target_h), (255, 255, 255))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        canvas.paste(img_resized, (paste_x, paste_y))
        _fmt = load_app_settings().get("output_format", "WEBP")
        _ext, _pil_fmt, _ = get_format_extension(_fmt)
        _save_kwargs = {"quality": 100, "subsampling": 0} if _pil_fmt == "JPEG" else \
                       {"quality": 100} if _pil_fmt in ("WEBP", "AVIF") else {}
        canvas.save(output_path, _pil_fmt, **_save_kwargs)
        return True
    except Exception:
        return False


# ===================== PIPELINE FUNCTION =====================
def process_single_image(image_path, filename, model, classes, conf_thresh, qual_thresh,
                         ocr_reader=None, file_size=None, watermark_logo=None,
                         enable_text_removal=True):
    """
    Process a single image through the complete pipeline
    Returns: dict with processing results
    """
    result = {
        "filename": filename,
        "status": "error",
        "category": "rejected",
        "category_raw": "rejected",
        "confidence": 0.0,
        "quality_score": 0.0,
        "quality_status": "Bad Quality",
        "output_path": None,
        "error": None
    }

    try:
        # Step 1: Classify image
        if model is None:
            result["error"] = "Model not loaded"
            return result

        label, conf, all_probs = predict_image(image_path, model, classes)
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        top_conf  = sorted_probs[0][1] if sorted_probs else 0
        top_label = sorted_probs[0][0] if sorted_probs else label

        # Determine category
        if top_conf < conf_thresh:
            try:
                _chk  = cv2.imread(image_path)
                _gray = cv2.cvtColor(_chk, cv2.COLOR_BGR2GRAY)
                _var  = cv2.Laplacian(_gray, cv2.CV_64F).var()
                category = top_label if (_var > 50 and top_conf > 0.30) else "rejected"
            except Exception:
                category = "rejected"
        else:
            category = top_label

        # Step 2: Assess quality
        quality_score, metrics = assess_image_quality(image_path)
        quality_status = "Good Quality" if quality_score >= qual_thresh else "Bad Quality"

        # Step 3: Get output path
        with Image.open(image_path) as img:
            width, height = img.size
        quality_folder = "good_quality" if quality_score >= qual_thresh else "bad_quality"
        output_dir = os.path.join(OUTPUT_DIR, category, quality_folder)
        os.makedirs(output_dir, exist_ok=True)

        base, ext = os.path.splitext(filename)
        _out_fmt = load_app_settings().get("output_format", "WEBP")
        _out_ext, _, _ = get_format_extension(_out_fmt)

        image_id = str(int(_time.time()))
        new_filename = f"{image_id}-{base}.{_out_ext}"
        output_path = os.path.join(output_dir, new_filename)
        counter = 1
        while os.path.exists(output_path):
            new_filename = f"{image_id}_{counter}-{base}.{_out_ext}"
            output_path = os.path.join(output_dir, new_filename)
            counter += 1

        temp_resized = None
        temp_cleaned = None

        # Step 4: Resize if needed
        if category != "rejected" and CATEGORY_SIZES[category]:
            _tmp_ext     = load_app_settings().get("output_format", "WEBP").lower()
            _tmp_ext     = "jpg" if _tmp_ext in ("jpeg", "jpg", "avif") else _tmp_ext
            temp_resized = os.path.join(TEMP_DIR, f"temp_resized_{image_id}.{_tmp_ext}")
            if resize_image(image_path, CATEGORY_SIZES[category], temp_resized):
                image_path = temp_resized

        # Step 5: Remove text if enabled
        if enable_text_removal and category != "rejected" and ocr_reader is not None:
            text_category_name = CATEGORY_TO_TEXT_REMOVAL.get(category)
            if text_category_name:
                _tmp_ext     = load_app_settings().get("output_format", "WEBP").lower()
                _tmp_ext     = "jpg" if _tmp_ext in ("jpeg", "jpg", "avif") else _tmp_ext
                temp_cleaned = os.path.join(TEMP_DIR, f"temp_cleaned_{image_id}.{_tmp_ext}")
                success, removed_pct, error = remove_text_and_logos(
                    image_path, temp_cleaned, text_category_name, ocr_reader
                )
                if success:
                    image_path = temp_cleaned

        # Step 6: Apply watermark and save
        apply_watermark_to_image(image_path, output_path, watermark_logo)

        # Step 7: Cleanup temp files
        if temp_resized and os.path.exists(temp_resized):
            os.remove(temp_resized)
        if temp_cleaned and os.path.exists(temp_cleaned):
            os.remove(temp_cleaned)

        result["status"] = "success"
        result["category"] = CATEGORY_MAPPING[category]
        result["category_raw"] = category
        result["confidence"] = round(top_conf, 4)
        result["quality_score"] = quality_score
        result["quality_status"] = quality_status
        result["output_path"] = output_path

    except Exception as e:
        result["error"] = str(e)

    return result


def predict_image(img_path: str, model, classes):
    """Predict image class"""
    if model is None:
        return 'error', 0.0, {}
    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        arr_div   = arr / 255.0
        arr_keras = (arr - 127.5) / 127.5
        def _run(a):
            return model.predict(np.expand_dims(a, 0), verbose=0)[0]
        preds_div   = _run(arr_div)
        preds_keras = _run(arr_keras)
        preds = preds_div if np.max(preds_div) >= np.max(preds_keras) else preds_keras
        idx        = int(np.argmax(preds))
        label      = classes[idx]
        confidence = float(preds[idx])
        all_probs  = {classes[i]: float(preds[i]) for i in range(len(classes))}
        return label, confidence, all_probs
    except Exception:
        return 'error', 0.0, {}


def assess_image_quality(image_path):
    """Comprehensive image quality assessment"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0, {"error": "Cannot read image"}

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Quick blur check
        _lap_pre = cv2.Laplacian(gray, cv2.CV_64F).var()
        if _lap_pre < 80:
            return round(max(0, (_lap_pre / 80) * 44), 2), {
                "overall": round(max(0, (_lap_pre / 80) * 44), 2),
                "sharpness": round((_lap_pre / 80) * 40, 2),
                "error": "Image too blurry"
            }

        # Basic quality metrics
        blur_score = min(100, max(0, (_lap_pre / 300) * 100))
        brightness = np.mean(gray)
        contrast = gray.std()
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size

        brightness_score = min(100, min(brightness, 255 - brightness) / 127.5 * 100)
        contrast_score = min(100, contrast / 100 * 100)
        sharpness_score = blur_score
        edge_score = min(100, edge_ratio * 2000)
        noise_score = 50

        quality_score = (
            sharpness_score  * 0.30 +
            blur_score       * 0.25 +
            contrast_score   * 0.15 +
            brightness_score * 0.12 +
            noise_score      * 0.10 +
            edge_score       * 0.05 +
            50               * 0.03
        )

        if brightness > 240 or brightness < 20:
            quality_score = min(quality_score, 30)
        if contrast < 10:
            quality_score = min(quality_score, 25)

        quality_score = round(min(100, max(0, quality_score)), 2)

        return quality_score, {
            "overall": quality_score,
            "sharpness": round(sharpness_score, 2),
            "blur": round(blur_score, 2),
            "brightness": round(brightness_score, 2),
            "contrast": round(contrast_score, 2),
            "lap_var": round(_lap_pre, 2),
        }

    except Exception as e:
        return 0, {"error": f"Processing failed: {str(e)}"}
