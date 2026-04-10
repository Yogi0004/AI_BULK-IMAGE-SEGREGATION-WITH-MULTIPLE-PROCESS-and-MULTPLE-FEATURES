"""
Homes247 - AI Image Cleaner (Streamlit Version)
Removes text, logos, legends & watermarks from:
  - Master Plan Images
  - Floor Plan Images
  - Gallery Images
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
import io

st.set_page_config(
    page_title="Homes247 | AI Image Cleaner",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0f0f1a; }
    [data-testid="stSidebar"]          { background: #1a1a2e; border-right: 1px solid #2d2d50; }
    .block-container                   { padding-top: 1.5rem; }
    h1, h2, h3, h4, .stMarkdown p     { color: #e0e7ff; }

    .card {
        background: #16213e;
        border: 1px solid #2d2d50;
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 18px;
    }
    .card-green  { border-left: 4px solid #10b981; }
    .card-red    { border-left: 4px solid #ef4444; }
    .card-blue   { border-left: 4px solid #3b82f6; }
    .card-purple { border-left: 4px solid #8b5cf6; }

    .metric-row { display: flex; gap: 12px; margin-bottom: 18px; }
    .metric-box {
        flex: 1; background: #1e1e3a; border-radius: 10px;
        padding: 14px 10px; text-align: center; border: 1px solid #2d2d50;
    }
    .metric-box .val { font-size: 1.7rem; font-weight: 700; color: #8b5cf6; }
    .metric-box .lbl { font-size: 0.75rem; color: #94a3b8; margin-top: 2px; }

    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6, #7c3aed);
        color: white; border: none; border-radius: 10px;
        padding: 0.6rem 1.4rem; font-weight: 600; font-size: 1rem;
        width: 100%; transition: 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #7c3aed, #6d28d9);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4);
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white; border: none; border-radius: 10px;
        padding: 0.55rem 1.2rem; font-weight: 600; width: 100%;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: #1e1e3a !important;
        border: 2px dashed #8b5cf6 !important;
        border-radius: 12px !important;
    }
    [data-testid="stImage"] img { border-radius: 10px; border: 1px solid #2d2d50; }

    .stTabs [data-baseweb="tab-list"]  { background: #1a1a2e; border-radius: 10px; padding: 4px; gap: 4px; }
    .stTabs [data-baseweb="tab"]       { background: transparent; color: #94a3b8; border-radius: 8px; padding: 8px 20px; font-weight: 600; }
    .stTabs [aria-selected="true"]     { background: #8b5cf6 !important; color: white !important; }

    .stCheckbox label { color: #e0e7ff !important; }
    [data-testid="stSelectbox"] > div > div {
        background: #1e1e3a; color: #e0e7ff;
        border: 1px solid #2d2d50; border-radius: 8px;
    }
    ::-webkit-scrollbar       { width: 6px; }
    ::-webkit-scrollbar-track { background: #0f0f1a; }
    ::-webkit-scrollbar-thumb { background: #8b5cf6; border-radius: 3px; }
    [data-testid="stSidebar"] label { color: #94a3b8 !important; }
    [data-testid="stSidebar"] .stMarkdown p { font-size: 0.85rem; }

    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600; margin: 2px;
    }
    .badge-green  { background: rgba(16,185,129,0.2); color: #10b981; border: 1px solid #10b981; }
    .badge-red    { background: rgba(239,68,68,0.2);  color: #ef4444; border: 1px solid #ef4444; }
    .badge-purple { background: rgba(139,92,246,0.2); color: #8b5cf6; border: 1px solid #8b5cf6; }
    .badge-blue   { background: rgba(59,130,246,0.2); color: #3b82f6; border: 1px solid #3b82f6; }
    hr { border-color: #2d2d50 !important; margin: 16px 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# KEYWORD SETS & SETTINGS  — keys MUST match exactly
# ─────────────────────────────────────────────
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

CATEGORY_OPTIONS = ["Master Plan", "Floor Plan", "Gallery"]


# ─────────────────────────────────────────────
# CACHED OCR READER
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI OCR engine…")
def load_ocr():
    try:
        return easyocr.Reader(['en'], gpu=False)
    except Exception as e:
        st.error(f"EasyOCR load error: {e}")
        return None


# ─────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────
def build_removal_mask(img_bgr, category, ocr_reader, extra_margin):
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    cfg      = SETTINGS[category]       # fixed: keys match exactly now
    keywords = KEYWORDS[category]
    ocr_margin = cfg["ocr_margin"] + extra_margin * 0.10

    # ── METHOD 1: OCR ──────────────────────────────────────────────────
    if ocr_reader is not None:
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            results = ocr_reader.readtext(gray)
            for det in results:
                bbox, text = det[0], det[1]
                pts = np.array(bbox, dtype=np.int32)
                cx  = np.mean(pts[:, 0])
                cy  = np.mean(pts[:, 1])
                at_edge   = (cx < w * ocr_margin or cx > w * (1 - ocr_margin) or
                             cy < h * ocr_margin or cy > h * (1 - ocr_margin))
                is_keyword = any(k in text.lower() for k in keywords)
                if at_edge or is_keyword:
                    exp = pts.copy()
                    exp[:, 0] = np.clip(pts[:, 0] + np.array([-10, 10, 10, -10]), 0, w)
                    exp[:, 1] = np.clip(pts[:, 1] + np.array([-10, -10, 10, 10]), 0, h)
                    cv2.fillPoly(mask, [exp.astype(np.int32)], 255)
        except Exception as e:
            st.warning(f"OCR issue: {e}")

    # ── METHOD 2: Corner & edge blanking ───────────────────────────────
    cx_pct = cfg["corner_pct"] + extra_margin * 0.05
    ey_pct = cfg["edge_pct"]   + extra_margin * 0.04
    cxs = int(w * cx_pct)
    cys = int(h * cx_pct)
    es  = int(min(h, w) * ey_pct)

    mask[0:cys, 0:cxs]       = 255
    mask[0:cys, w-cxs:w]     = 255
    mask[h-cys:h, 0:cxs]     = 255
    mask[h-cys:h, w-cxs:w]   = 255
    mask[0:es, :]             = 255
    mask[h-es:h, :]           = 255
    mask[:, 0:es]             = 255
    mask[:, w-es:w]           = 255

    # ── METHOD 3: Isolated contour detection ───────────────────────────
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area  = h * w
    edge_check  = 0.22

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, cw, ch = cv2.boundingRect(cnt)
        near_edge = (x < w * edge_check or x + cw > w * (1 - edge_check) or
                     y < h * edge_check or y + ch > h * (1 - edge_check))
        is_small      = area < total_area * 0.04
        aspect        = max(cw, ch) / (min(cw, ch) + 1)
        is_text_shape = aspect > 3.5
        if near_edge and (is_small or is_text_shape):
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            xp = max(5, int(cw * 0.1))
            yp = max(5, int(ch * 0.1))
            cv2.rectangle(mask,
                          (max(0, x - xp), max(0, y - yp)),
                          (min(w, x + cw + xp), min(h, y + ch + yp)), 255, -1)

    # ── METHOD 4: Legend box detection (plan images only) ──────────────
    if category in ("Master Plan", "Floor Plan"):
        edges_det = cv2.Canny(gray, 50, 150)
        kr     = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        closed = cv2.morphologyEx(edges_det, cv2.MORPH_CLOSE, kr)
        rc, _  = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in rc:
            x, y, cw, ch = cv2.boundingRect(cnt)
            a          = cw * ch
            is_legend  = 0.01 < (a / total_area) < 0.15
            near       = (x < w * 0.25 or x + cw > w * 0.75 or
                          y < h * 0.25 or y + ch > h * 0.75)
            asp        = max(cw, ch) / (min(cw, ch) + 1)
            if is_legend and near and 1 < asp < 4:
                cv2.rectangle(mask, (x - 10, y - 10),
                              (x + cw + 10, y + ch + 10), 255, -1)

    # ── Finalise ───────────────────────────────────────────────────────
    k    = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, k, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def find_main_plan_area(img_bgr):
    h, w   = img_bgr.shape[:2]
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges  = cv2.Canny(gray, 30, 100)
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
    kp        = np.ones((10, 10), np.uint8)
    main_mask = cv2.dilate(main_mask, kp, iterations=1)
    return main_mask


def clean_image(pil_image, category, ocr_reader,
                protect_center, extra_margin, inpaint_radius):
    img_arr = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)

    mask = build_removal_mask(img_bgr, category, ocr_reader, extra_margin)

    if protect_center:
        main_area = find_main_plan_area(img_bgr)
        mask      = cv2.bitwise_and(mask, cv2.bitwise_not(main_area))

    result      = cv2.inpaint(img_bgr, mask, inpaintRadius=inpaint_radius,
                               flags=cv2.INPAINT_TELEA)
    removed_pct = round(np.sum(mask == 255) / mask.size * 100, 2)
    result_pil  = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    mask_pil    = Image.fromarray(mask)
    return result_pil, mask_pil, removed_pct


def pil_to_bytes(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0 20px;'>
        <div style='font-size:2.2rem;'>🏠</div>
        <div style='font-size:1.2rem; font-weight:700; color:#8b5cf6;'>Homes247</div>
        <div style='font-size:0.78rem; color:#94a3b8;'>AI Image Cleaner</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='color:#94a3b8; font-size:0.8rem; margin:0 0 10px;'>⚙️ PROCESSING SETTINGS</p>",
                unsafe_allow_html=True)

    protect = st.checkbox("🛡️ Protect Central Area", value=True,
                          help="Prevents removal of main plan content in center")

    extra_margin = st.slider("🔍 Detection Aggressiveness", 0, 5, 2,
                             help="Higher = removes more peripheral content")

    inpaint_radius = st.slider("🎨 Inpaint Smoothness", 1, 8, 3,
                               help="Radius used to fill removed regions")

    show_mask = st.checkbox("🎭 Show Removal Mask", value=True,
                            help="Display which areas were detected and removed")

    st.markdown("---")
    st.markdown("""
    <div class='card card-green'>
        <p style='color:#10b981; font-weight:700; margin:0 0 6px;'>✅ What's Preserved</p>
        <p style='color:#94a3b8; font-size:0.82rem; margin:0;'>
        • Building layouts & plans<br>
        • Unit / room labels inside<br>
        • Parking & amenity labels<br>
        • Gallery photo content<br>
        • All architectural drawings
        </p>
    </div>
    <div class='card card-red'>
        <p style='color:#ef4444; font-weight:700; margin:0 0 6px;'>🗑️ What Gets Removed</p>
        <p style='color:#94a3b8; font-size:0.82rem; margin:0;'>
        • Company logos (all corners)<br>
        • MASTER / FLOOR PLAN titles<br>
        • Website URLs & watermarks<br>
        • Legend & scale boxes<br>
        • North arrows & direction text<br>
        • Copyright / disclaimer text
        </p>
    </div>
    <div class='card card-blue'>
        <p style='color:#3b82f6; font-weight:700; margin:0 0 6px;'>🔬 AI Pipeline</p>
        <p style='color:#94a3b8; font-size:0.82rem; margin:0;'>
        1. OCR text detection<br>
        2. Corner / edge blanking<br>
        3. Contour analysis<br>
        4. Legend box detection<br>
        5. Smart inpainting
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:0 0 24px;'>
    <h1 style='font-size:2.4rem; font-weight:800; color:#8b5cf6; margin:0;'>
        🏠 Homes247 AI Image Cleaner
    </h1>
    <p style='color:#94a3b8; font-size:1.05rem; margin:6px 0 0;'>
        Removes text, logos, legends & watermarks — instantly.<br>
        Works on <b style='color:#e0e7ff;'>Master Plans · Floor Plans · Gallery Images</b>
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD OCR
# ─────────────────────────────────────────────
ocr_reader = load_ocr()
if ocr_reader:
    st.success("✅ AI OCR Engine Ready", icon="🤖")
else:
    st.warning("⚠️ OCR unavailable — edge-based removal only", icon="⚠️")


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_single, tab_batch = st.tabs(["🖼️  Single Image", "📦  Batch Upload"])


# ══════════════════════════════════════════════
# SINGLE IMAGE TAB
# ══════════════════════════════════════════════
with tab_single:

    st.markdown("""
    <div class='card card-purple'>
        <p style='color:#8b5cf6; font-weight:700; font-size:1.1rem; margin:0 0 4px;'>
            🧹 AI Text & Logo Remover
        </p>
        <p style='color:#94a3b8; font-size:0.88rem; margin:0;'>
            Upload any real estate image and select its type — the AI will remove all
            unwanted text, logos, watermarks and legends automatically.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload + category in one row ──────────────────────────────────
    up_col, cat_col = st.columns([3, 1], gap="medium")

    with up_col:
        uploaded = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png", "webp"],
            key="single_upload",
            label_visibility="collapsed",
        )

    with cat_col:
        st.markdown("<p style='color:#94a3b8; font-size:0.85rem; margin-bottom:6px;'>📂 Image Type</p>",
                    unsafe_allow_html=True)
        category = st.selectbox(
            "Image Type",
            options=CATEGORY_OPTIONS,          # ["Master Plan", "Floor Plan", "Gallery"]
            key="single_cat",
            label_visibility="collapsed",
        )

    # ── Process ───────────────────────────────────────────────────────
    if uploaded:
        pil_img        = Image.open(uploaded).convert("RGB")
        w_orig, h_orig = pil_img.size

        col_orig, col_result = st.columns(2, gap="medium")

        with col_orig:
            st.markdown("<p style='color:#94a3b8; font-size:0.85rem; margin-bottom:6px;'>📤 ORIGINAL</p>",
                        unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True)
            st.markdown(f"""
            <div class='metric-row'>
                <div class='metric-box'>
                    <div class='val'>{w_orig}</div><div class='lbl'>Width (px)</div>
                </div>
                <div class='metric-box'>
                    <div class='val'>{h_orig}</div><div class='lbl'>Height (px)</div>
                </div>
                <div class='metric-box'>
                    <div class='val'>{round(uploaded.size/1024,1)}</div><div class='lbl'>Size (KB)</div>
                </div>
            </div>
            <div style='margin-bottom:10px;'>
                <span class='badge badge-purple'>Type: {category}</span>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🚀 Remove Text & Logos", key="single_btn"):
            with st.spinner(f"Processing {category} image…"):
                result_img, mask_img, removed_pct = clean_image(
                    pil_img, category, ocr_reader,
                    protect, extra_margin, inpaint_radius
                )
            st.session_state["s_result"]  = result_img
            st.session_state["s_mask"]    = mask_img
            st.session_state["s_removed"] = removed_pct

        # ── Show results ──────────────────────────────────────────────
        if "s_result" in st.session_state:
            result_img  = st.session_state["s_result"]
            mask_img    = st.session_state["s_mask"]
            removed_pct = st.session_state["s_removed"]

            with col_result:
                st.markdown("<p style='color:#10b981; font-size:0.85rem; margin-bottom:6px;'>✨ CLEANED</p>",
                            unsafe_allow_html=True)
                st.image(result_img, use_container_width=True)
                st.markdown(f"""
                <div class='metric-row'>
                    <div class='metric-box'>
                        <div class='val' style='color:#10b981;'>{removed_pct}%</div>
                        <div class='lbl'>Area Cleaned</div>
                    </div>
                    <div class='metric-box'>
                        <div class='val' style='color:#10b981;'>✓</div>
                        <div class='lbl'>Center Protected</div>
                    </div>
                    <div class='metric-box'>
                        <div class='val' style='color:#10b981;'>AI</div>
                        <div class='lbl'>OCR + CV</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if show_mask:
                st.markdown("---")
                st.markdown("<p style='color:#94a3b8; font-size:0.85rem;'>🎭 REMOVAL MASK — white areas were detected & removed</p>",
                            unsafe_allow_html=True)
                st.image(mask_img, use_container_width=True)

            st.markdown("---")
            dl1, dl2 = st.columns(2, gap="small")
            with dl1:
                st.download_button(
                    label="⬇️ Download Cleaned Image",
                    data=pil_to_bytes(result_img, "JPEG"),
                    file_name=f"homes247_cleaned_{category.lower().replace(' ', '_')}.jpg",
                    mime="image/jpeg",
                    key="dl_result_single",
                )
            with dl2:
                st.download_button(
                    label="⬇️ Download Removal Mask",
                    data=pil_to_bytes(mask_img, "PNG"),
                    file_name=f"homes247_mask_{category.lower().replace(' ', '_')}.png",
                    mime="image/png",
                    key="dl_mask_single",
                )

    else:
        st.markdown("""
        <div style='text-align:center; padding:60px 20px; color:#4a4a7a;'>
            <div style='font-size:3.5rem;'>⬆️</div>
            <p style='font-size:1.15rem; margin-top:12px; color:#6a6a9a;'>
                Upload an image above and select its type to get started
            </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# BATCH TAB
# ══════════════════════════════════════════════
with tab_batch:
    st.markdown("""
    <div class='card card-blue'>
        <p style='color:#3b82f6; font-weight:700; font-size:1.1rem; margin:0 0 4px;'>
            📦 Batch Image Cleaner
        </p>
        <p style='color:#94a3b8; font-size:0.88rem; margin:0;'>
            Upload multiple images at once. Select the category that applies to all of them.
        </p>
    </div>
    """, unsafe_allow_html=True)

    bup_col, bcat_col = st.columns([3, 1], gap="medium")

    with bup_col:
        batch_files = st.file_uploader(
            "Upload Multiple Images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="batch_upload",
            label_visibility="collapsed",
        )

    with bcat_col:
        st.markdown("<p style='color:#94a3b8; font-size:0.85rem; margin-bottom:6px;'>📂 Image Type</p>",
                    unsafe_allow_html=True)
        batch_cat = st.selectbox(
            "Batch Image Type",
            options=CATEGORY_OPTIONS,
            key="batch_cat",
            label_visibility="collapsed",
        )

    if batch_files:
        st.markdown(f"""
        <div style='margin-bottom:14px;'>
            <span class='badge badge-purple'>📂 {len(batch_files)} files selected</span>
            <span class='badge badge-blue'>Type: {batch_cat}</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Process All Images", key="batch_btn"):
            progress = st.progress(0, text="Starting…")
            results  = {}
            total    = len(batch_files)

            for i, f in enumerate(batch_files):
                progress.progress(i / total, text=f"Processing {f.name} ({i+1}/{total})…")
                try:
                    pil_img = Image.open(f).convert("RGB")
                    res_img, msk_img, rpct = clean_image(
                        pil_img, batch_cat, ocr_reader,
                        protect, extra_margin, inpaint_radius
                    )
                    results[f.name] = {
                        "original": pil_img,
                        "result":   res_img,
                        "mask":     msk_img,
                        "removed":  rpct,
                        "status":   "✅ Success",
                    }
                except Exception as e:
                    results[f.name] = {"status": f"❌ Error: {e}"}

            progress.progress(1.0, text="All done!")
            st.session_state["batch_results"] = results

        if "batch_results" in st.session_state:
            res = st.session_state["batch_results"]
            st.markdown("---")

            ok  = sum(1 for v in res.values() if "result" in v)
            err = len(res) - ok
            avg = round(sum(v["removed"] for v in res.values() if "removed" in v) / max(ok, 1), 2)

            st.markdown(f"""
            <div class='metric-row'>
                <div class='metric-box'>
                    <div class='val' style='color:#10b981;'>{ok}</div>
                    <div class='lbl'>Processed OK</div>
                </div>
                <div class='metric-box'>
                    <div class='val' style='color:#ef4444;'>{err}</div>
                    <div class='lbl'>Errors</div>
                </div>
                <div class='metric-box'>
                    <div class='val' style='color:#8b5cf6;'>{avg}%</div>
                    <div class='lbl'>Avg. Cleaned</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            for fname, data in res.items():
                with st.expander(f"{data['status']}  —  {fname}"):
                    if "result" in data:
                        c1, c2 = st.columns(2, gap="small")
                        with c1:
                            st.markdown("<p style='color:#94a3b8; font-size:0.8rem;'>ORIGINAL</p>",
                                        unsafe_allow_html=True)
                            st.image(data["original"], use_container_width=True)
                        with c2:
                            st.markdown(f"<p style='color:#10b981; font-size:0.8rem;'>CLEANED — {data['removed']}% removed</p>",
                                        unsafe_allow_html=True)
                            st.image(data["result"], use_container_width=True)
                        st.download_button(
                            label=f"⬇️ Download {fname}",
                            data=pil_to_bytes(data["result"], "JPEG"),
                            file_name=f"cleaned_{fname.rsplit('.', 1)[0]}.jpg",
                            mime="image/jpeg",
                            key=f"batch_dl_{fname}",
                        )
                    else:
                        st.error(data["status"])
    else:
        st.markdown("""
        <div style='text-align:center; padding:60px 20px; color:#4a4a7a;'>
            <div style='font-size:3.5rem;'>📂</div>
            <p style='font-size:1.15rem; margin-top:12px; color:#6a6a9a;'>
                Upload multiple images above to batch-process them
            </p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#4a4a7a; font-size:0.82rem; padding:10px 0 4px;'>
    🏠 <b style='color:#8b5cf6;'>Homes247</b> — India's Favourite Property Portal &nbsp;|&nbsp;
    AI Image Cleaner v2.0 &nbsp;|&nbsp; Powered by EasyOCR + OpenCV
</div>
""", unsafe_allow_html=True)