"""
=============================================================================
  Homes247 — Blur Detection Streamlit Dashboard
  blur_dashboard.py
=============================================================================

HOW TO RUN:
    streamlit run blur_dashboard.py

REQUIRES:
    train.py must be in the same folder.

WHAT THIS DASHBOARD DOES:
    Tab 1 — Single Image Test  : upload any image, get full quality breakdown
    Tab 2 — Blur Simulator     : drag a slider to see how blur affects the score
    Tab 3 — Batch Folder Test  : test every image in a folder, see Good/Bad split
    Tab 4 — Train Model        : run training right from the UI
    Tab 5 — Score Guide        : understand what each score means
=============================================================================
"""

import os, json, time, tempfile, warnings
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image, ImageFilter

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ── Import training module ────────────────────────────────────────────────────
try:
    import cv2
    from train_resizer import assess_image_quality
    

    assess_image_quality = train.assess_image_quality
    generate_dataset = train.generate_dataset
    train_model = train.train_model

    GOOD_THRESHOLD = train.GOOD_THRESHOLD
    BASE_DIR = train.BASE_DIR
    DATASET_DIR = train.DATASET_DIR
    MODEL_SAVE_PATH = train.MODEL_SAVE_PATH
    IMG_SIZE = train.IMG_SIZE

    _TRAIN_OK = True
except ImportError as e:
    _TRAIN_OK = False
    _IMPORT_ERR = str(e)


# =============================================================================
#  PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Homes247 · Blur Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
#  CSS
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0d0d18;
}
section[data-testid="stSidebar"] {
    background: #0a0a15 !important;
    border-right: 1px solid #1e1e35 !important;
}

/* ── Metric cards ── */
.kpi-card {
    background: #13132a;
    border: 1px solid #252550;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.kpi-value {
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1.1;
    margin: 0.3rem 0;
    font-family: 'DM Mono', monospace;
}
.kpi-label {
    font-size: 0.78rem;
    color: #55557a;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Score ring (big number) ── */
.score-ring {
    width: 160px;
    height: 160px;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin: 0 auto 1rem auto;
    border: 5px solid;
}

/* ── Metric bar ── */
.mbar-wrap { margin-bottom: 9px; display: flex; align-items: center; gap: 10px; }
.mbar-name { width: 130px; font-size: 0.80rem; color: #7070a0; flex-shrink: 0; }
.mbar-bg   { flex: 1; height: 7px; background: #1c1c38; border-radius: 4px; overflow: hidden; }
.mbar-fill { height: 100%; border-radius: 4px; }
.mbar-val  { width: 42px; text-align: right; font-family: 'DM Mono', monospace;
             font-size: 0.80rem; color: #c0c0e0; flex-shrink: 0; }

/* ── Section pill ── */
.sec-pill {
    display: inline-block;
    background: #6366f120;
    border: 1px solid #6366f140;
    color: #818cf8;
    border-radius: 999px;
    padding: 0.3rem 1rem;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

/* ── Verdict badge ── */
.verdict { border-radius: 10px; padding: 0.6rem 1.2rem; font-weight: 700;
           font-size: 1rem; text-align: center; margin-top: 0.8rem; }

/* ── Debug table row ── */
.drow { display: flex; justify-content: space-between; margin-bottom: 4px;
        font-size: 0.78rem; }
.dkey { color: #55557a; }
.dval { color: #c0c0e0; font-family: 'DM Mono', monospace; }

/* ── Batch result pill ── */
.bpill-good { background:#10b98120; border:1px solid #10b98140; color:#10b981;
              border-radius:6px; padding:2px 8px; font-size:0.78rem; font-weight:600; }
.bpill-bad  { background:#ef444420; border:1px solid #ef444440; color:#ef4444;
              border-radius:6px; padding:2px 8px; font-size:0.78rem; font-weight:600; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: transform .2s, box-shadow .2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px #6366f155 !important;
}

/* ── Progress bar colour ── */
.stProgress > div > div { background: linear-gradient(90deg,#6366f1,#10b981) !important; }

/* ── Selectbox / slider accent ── */
[data-baseweb="select"] > div  { background: #13132a !important; border-color: #252550 !important; }
[data-testid="stSlider"] > div > div > div { background: #6366f1 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #13132a !important;
    border: 2px dashed #252550 !important;
    border-radius: 12px !important;
}

h1,h2,h3,h4 { color: #e0e0ff !important; }
p, li { color: #8080a0; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  HELPERS
# =============================================================================

QUALITY_TABLE = [
    (85, 100, "Crystal Clear",  "#22d3ee"),
    (70,  85, "Sharp",          "#10b981"),
    (55,  70, "Good",           "#84cc16"),
    (40,  55, "Acceptable",     "#f59e0b"),
    (25,  40, "Blurry",         "#f97316"),
    (10,  25, "Very Blurry",    "#ef4444"),
    ( 0,  10, "Unusable",       "#991b1b"),
]

def quality_info(score: float):
    for lo, hi, label, color in QUALITY_TABLE:
        if lo <= score <= hi:
            return label, color
    return "Unusable", "#991b1b"


def metric_bar(name: str, value: float, color: str):
    v = min(max(value, 0), 100)
    st.markdown(f"""
<div class="mbar-wrap">
  <div class="mbar-name">{name}</div>
  <div class="mbar-bg">
    <div class="mbar-fill" style="width:{v:.1f}%; background:{color};"></div>
  </div>
  <div class="mbar-val">{v:.1f}</div>
</div>""", unsafe_allow_html=True)


def pil_blur(img: Image.Image, level: int) -> Image.Image:
    return img if level == 0 else img.filter(ImageFilter.GaussianBlur(radius=level * 1.4))


def tmp_save(img: Image.Image) -> str:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    img.save(f.name, quality=95)
    return f.name


def verdict_html(score, threshold):
    good   = score >= threshold
    text   = "✅  GOOD QUALITY" if good else "❌  BAD QUALITY"
    bg     = "#10b98118" if good else "#ef444418"
    border = "#10b98150" if good else "#ef444450"
    color  = "#10b981"   if good else "#ef4444"
    return (f"<div class='verdict' style='background:{bg}; border:1px solid "
            f"{border}; color:{color};'>{text}</div>")


# =============================================================================
#  SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("""
<div style="background:linear-gradient(135deg,#6366f1,#7c3aed);
     border-radius:14px; padding:1.2rem; text-align:center; margin-bottom:1.5rem;">
  <div style="font-size:2rem;">🔬</div>
  <div style="color:#fff; font-weight:700; font-size:1.05rem; margin-top:.3rem;">
    Blur Detection
  </div>
  <div style="color:#c4b5fd; font-size:.78rem;">Homes247 AI Quality System</div>
</div>
""", unsafe_allow_html=True)

    if not _TRAIN_OK:
        st.error(f"❌ train.py import failed:\n{_IMPORT_ERR}")
        st.stop()

    # Model path
    st.markdown("### ⚙️ Model")
    model_path  = st.text_input("blur_model.h5 path", value=MODEL_SAVE_PATH)
    model_ok    = os.path.exists(model_path)

    if model_ok:
        msize = os.path.getsize(model_path) / 1024 / 1024
        st.success(f"✅ Model loaded — {msize:.1f} MB")
        meta_path = model_path.replace(".h5", "_metadata.json")
        if os.path.exists(meta_path):
            m = json.load(open(meta_path))
            st.markdown(f"""
<div style="background:#10b98110; border:1px solid #10b98130; border-radius:10px;
     padding:.8rem; font-size:.80rem; margin-top:.4rem;">
  <span style="color:#10b981; font-weight:600;">Last training</span><br/>
  <span style="color:#80ffc0;">
    Accuracy {m.get('test_accuracy_pct',0):.1f}% &nbsp;|&nbsp;
    F1 {m.get('test_f1_pct',0):.1f}%<br/>
    MAE {m.get('test_mae',0):.4f} &nbsp;|&nbsp;
    {m.get('trained_at','')[:10]}
  </span>
</div>""", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Model not found — CV-only mode")

    model_weight = st.slider(
        "AI blend weight",
        0.0, 0.9, 0.60, 0.05,
        help="Higher = trust AI model more vs CV algorithms",
        disabled=not model_ok,
    )

    st.markdown("---")

    # Quality threshold
    st.markdown("### 🎯 Threshold")
    _QUAL_OPTS = {
        "Very Low  (10)  — Extremely blurry": 10,
        "Low       (25)  — Very blurry":      25,
        "Medium    (40)  — Partially visible":40,
        "Good      (55)  — Content visible ✅":55,
        "High      (70)  — Clearly visible":  70,
        "Very High (85)  — Crystal clear":    85,
    }
    sel_q      = st.selectbox("Quality threshold", list(_QUAL_OPTS.keys()), index=3)
    threshold  = _QUAL_OPTS[sel_q]

    st.markdown(f"""
<div style="background:#6366f110; border:1px solid #6366f130; border-radius:10px;
     padding:.8rem; margin-top:.4rem; font-size:.82rem;">
  <span style="color:#818cf8;">
    Score ≥ <b>{threshold}</b> → ✅ Good Quality<br/>
    Score &lt; <b>{threshold}</b> → ❌ Bad  Quality
  </span>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Category override
    st.markdown("### 🖼️ Image Type")
    category = st.selectbox(
        "Category (auto = detect)",
        ["auto", "gallery", "floorplan", "masterplan"],
    )


# =============================================================================
#  HEADER
# =============================================================================
st.markdown("""
<div style="background:linear-gradient(135deg,#6366f112,#10b98108);
     border:1px solid #6366f130; border-radius:18px;
     padding:1.8rem 2rem; margin-bottom:1.8rem; display:flex;
     align-items:center; gap:1.5rem;">
  <div style="font-size:3rem;">🔬</div>
  <div>
    <h1 style="margin:0; font-size:1.9rem; background:linear-gradient(135deg,#818cf8,#10b981);
       -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
      Blur Detection Dashboard
    </h1>
    <p style="margin:.3rem 0 0 0; color:#55557a; font-size:.9rem;">
      AI-powered image quality analysis · Floor Plans · Master Plans · Gallery
    </p>
  </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
#  TABS
# =============================================================================
t1, t2, t3, t4, t5 = st.tabs([
    "🖼️ Single Image",
    "🎛️ Blur Simulator",
    "📁 Batch Folder",
    "🎓 Train Model",
    "📊 Score Guide",
])


# =============================================================================
#  TAB 1 — SINGLE IMAGE TEST
# =============================================================================
with t1:
    st.markdown('<div class="sec-pill">Upload an image → get full quality breakdown</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop image here", type=["jpg","jpeg","png","webp"],
                                key="single_up")

    if uploaded:
        pil = Image.open(uploaded).convert("RGB")
        tmp = tmp_save(pil)

        with st.spinner("Analysing…"):
            score, metrics = assess_image_quality(
                tmp, category=category,
                model_path=model_path if model_ok else None,
                model_weight=model_weight,
            )
        try: os.unlink(tmp)
        except: pass

        ql, qc = quality_info(score)
        is_good = score >= threshold

        # ── Row: image | score ring | metrics ─────────────────────────────
        c_img, c_score, c_metrics = st.columns([2.0, 1.2, 1.8])

        with c_img:
            st.image(pil, use_container_width=True, caption=uploaded.name)
            w, h = pil.size
            ai_s = metrics.get("ai_score")
            ai_txt = f" | AI {ai_s:.1f}" if ai_s is not None else " | CV only"
            st.caption(f"{w}×{h}px · {uploaded.size/1024:.1f}KB · "
                       f"{metrics.get('category_used','?').upper()}{ai_txt}")

        with c_score:
            st.markdown(f"""
<div class="score-ring" style="border-color:{qc}; background:{qc}12;">
  <div style="color:#9090b0; font-size:.72rem; text-transform:uppercase;
       letter-spacing:.08em;">Score</div>
  <div class="kpi-value" style="color:{qc}; font-size:3rem;">{score:.1f}</div>
  <div style="color:#55557a; font-size:.72rem;">/100</div>
</div>
<div style="text-align:center; color:{qc}; font-weight:700;
     font-size:1.05rem; margin-bottom:.5rem;">{ql}</div>
""", unsafe_allow_html=True)
            st.markdown(verdict_html(score, threshold), unsafe_allow_html=True)

            # AI / CV split
            if metrics.get("ai_score") is not None:
                st.markdown(f"""
<div style="background:#13132a; border:1px solid #252550; border-radius:10px;
     padding:.8rem; margin-top:.7rem; font-size:.80rem;">
  <div style="color:#818cf8; font-weight:600; margin-bottom:.4rem;">Score split</div>
  <div style="display:flex; justify-content:space-between;">
    <span style="color:#7070a0;">🤖 AI</span>
    <span style="color:#818cf8; font-family:'DM Mono',monospace;">
      {metrics['ai_score']:.1f}
    </span>
  </div>
  <div style="display:flex; justify-content:space-between; margin-top:.3rem;">
    <span style="color:#7070a0;">📐 CV</span>
    <span style="color:#10b981; font-family:'DM Mono',monospace;">
      {metrics['cv_score']:.1f}
    </span>
  </div>
</div>""", unsafe_allow_html=True)

        with c_metrics:
            st.markdown('<div style="color:#818cf8; font-weight:600; '
                        'margin-bottom:.8rem;">Metric breakdown</div>',
                        unsafe_allow_html=True)
            metric_bar("🔬 Sharpness",     metrics.get("sharpness",    0), "#818cf8")
            metric_bar("⚡ Tenengrad",     metrics.get("tenengrad",    0), "#6366f1")
            metric_bar("🔲 Block content", metrics.get("block_content",0), "#8b5cf6")
            metric_bar("📏 Edge visible",  metrics.get("edge_visible", 0), "#a78bfa")
            metric_bar("🌓 Contrast",      metrics.get("contrast",     0), "#c4b5fd")

            with st.expander("🔧 Raw debug values"):
                for k, v in {
                    "laplacian_raw": metrics.get("laplacian_raw",0),
                    "edge_density":  metrics.get("edge_density", 0),
                    "contrast_raw":  metrics.get("contrast_raw",  0),
                    "block_ratio":   metrics.get("block_ratio",   0),
                    "cv_score":      metrics.get("cv_score",      0),
                    "cat_score":     metrics.get("cat_score",     0),
                    "ai_score":      metrics.get("ai_score"),
                    "category_used": metrics.get("category_used","?"),
                }.items():
                    vstr = f"{v:.4f}" if isinstance(v, float) else str(v)
                    st.markdown(
                        f'<div class="drow"><span class="dkey">{k}</span>'
                        f'<span class="dval">{vstr}</span></div>',
                        unsafe_allow_html=True)


# =============================================================================
#  TAB 2 — BLUR SIMULATOR
# =============================================================================
with t2:
    st.markdown('<div class="sec-pill">Drag the slider to see how blur affects the score</div>',
                unsafe_allow_html=True)

    sim_up = st.file_uploader("Upload image", type=["jpg","jpeg","png","webp"],
                               key="sim_up")

    if sim_up:
        sim_pil   = Image.open(sim_up).convert("RGB")
        blur_lvl  = st.slider("Blur level", 0, 18, 0)
        blurred   = pil_blur(sim_pil, blur_lvl)

        tmp_b = tmp_save(blurred)
        s_b, m_b = assess_image_quality(
            tmp_b, category=category,
            model_path=model_path if model_ok else None,
        )
        try: os.unlink(tmp_b)
        except: pass

        ql_b, qc_b = quality_info(s_b)

        c1, c2 = st.columns([2.5, 1.5])
        with c1:
            st.image(blurred, use_container_width=True,
                     caption=f"Blur level: {blur_lvl}")

        with c2:
            st.markdown(f"""
<div style="text-align:center; background:#13132a; border:1px solid #252550;
     border-radius:14px; padding:1.5rem; margin-bottom:1rem;">
  <div style="color:#55557a; font-size:.78rem; text-transform:uppercase;
       letter-spacing:.08em;">Quality Score</div>
  <div style="font-size:3.8rem; font-weight:700; color:{qc_b};
       font-family:'DM Mono',monospace; line-height:1.1;">{s_b:.1f}</div>
  <div style="color:#55557a; font-size:.75rem;">/100</div>
  <div style="color:{qc_b}; font-weight:700; margin-top:.5rem;">{ql_b}</div>
</div>""", unsafe_allow_html=True)

            st.markdown(verdict_html(s_b, threshold), unsafe_allow_html=True)
            st.markdown("<br/>", unsafe_allow_html=True)
            metric_bar("Sharpness",     m_b.get("sharpness",    0), "#818cf8")
            metric_bar("Tenengrad",     m_b.get("tenengrad",    0), "#6366f1")
            metric_bar("Block content", m_b.get("block_content",0), "#8b5cf6")
            metric_bar("Edge visible",  m_b.get("edge_visible", 0), "#a78bfa")
            metric_bar("Contrast",      m_b.get("contrast",     0), "#c4b5fd")

        # ── Score at all standard blur levels ────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="sec-pill">Score at all synthetic blur levels</div>',
                    unsafe_allow_html=True)
        cols = st.columns(8)
        for col, (ks, sigma, ref_score, label) in zip(cols, [
            (0,0.0,"Perfect"), (3,0.8,"Excellent"), (5,1.2,"Good"),
            (9,2.0,"Accept."), (13,3.0,"Poor"), (19,5.0,"Bad"),
            (27,8.0,"V.Bad"), (41,12.0,"Terrible")
        ]):
            lv = ks if isinstance(ks, int) else 0
            bl = pil_blur(sim_pil, max(0, lv // 3))
            tmp_l = tmp_save(bl)
            s_l, _ = assess_image_quality(tmp_l, category=category,
                                           model_path=model_path if model_ok else None)
            try: os.unlink(tmp_l)
            except: pass
            _, c_l = quality_info(s_l)
            with col:
                st.image(bl, use_container_width=True)
                st.markdown(f"""
<div style="text-align:center; font-size:.80rem;">
  <div style="color:{c_l}; font-weight:700; font-family:'DM Mono',monospace;">
    {s_l:.1f}
  </div>
  <div style="color:#55557a; font-size:.72rem;">{label}</div>
</div>""", unsafe_allow_html=True)


# =============================================================================
#  TAB 3 — BATCH FOLDER TEST
# =============================================================================
with t3:
    st.markdown('<div class="sec-pill">Test all images in a folder</div>',
                unsafe_allow_html=True)

    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        batch_dir  = st.text_input("Folder path", value=BASE_DIR)
    with col_in2:
        max_imgs   = st.number_input("Max images", 1, 1000, 50)
    with col_in3:
        sort_order = st.selectbox("Sort by", ["Score ↑", "Score ↓", "Filename"])

    show_prev  = st.checkbox("Show image thumbnails", value=True)

    if st.button("▶  Run Batch Analysis", use_container_width=True):
        if not os.path.isdir(batch_dir):
            st.error(f"Folder not found: {batch_dir}")
        else:
            exts   = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            paths  = [p for p in Path(batch_dir).iterdir()
                      if p.suffix.lower() in exts][:int(max_imgs)]

            if not paths:
                st.warning("No images found.")
            else:
                prog    = st.progress(0)
                status  = st.empty()
                results = []

                for i, p in enumerate(paths):
                    status.markdown(
                        f"<div style='color:#818cf8; font-size:.85rem;'>"
                        f"Analysing {i+1}/{len(paths)}: {p.name}</div>",
                        unsafe_allow_html=True,
                    )
                    score, metrics = assess_image_quality(
                        str(p), category=category,
                        model_path=model_path if model_ok else None,
                    )
                    results.append({
                        "path":     str(p),
                        "name":     p.name,
                        "score":    score,
                        "verdict":  "Good" if score >= threshold else "Bad",
                        "category": metrics.get("category_used", "?"),
                        "ai":       metrics.get("ai_score"),
                        "cv":       metrics.get("cv_score", score),
                    })
                    prog.progress((i + 1) / len(paths))

                status.empty()
                prog.empty()

                # Sort
                if sort_order == "Score ↑":
                    results.sort(key=lambda x: x["score"])
                elif sort_order == "Score ↓":
                    results.sort(key=lambda x: x["score"], reverse=True)
                else:
                    results.sort(key=lambda x: x["name"])

                # Summary KPIs
                n_good = sum(1 for r in results if r["verdict"] == "Good")
                n_bad  = len(results) - n_good
                avg_s  = np.mean([r["score"] for r in results])

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total",        len(results))
                k2.metric("✅ Good",      n_good, f"{n_good/len(results)*100:.1f}%")
                k3.metric("❌ Bad",       n_bad,  f"{n_bad/len(results)*100:.1f}%")
                k4.metric("Avg score",    f"{avg_s:.1f}")

                st.markdown("---")

                if show_prev:
                    COLS = 4
                    for row_i in range(0, len(results), COLS):
                        row_res = results[row_i:row_i+COLS]
                        cols    = st.columns(COLS)
                        for col, r in zip(cols, row_res):
                            with col:
                                try:
                                    st.image(r["path"], use_container_width=True)
                                except:
                                    st.write("⚠️ load error")
                                _, c_r = quality_info(r["score"])
                                pill = ("bpill-good" if r["verdict"]=="Good"
                                        else "bpill-bad")
                                st.markdown(f"""
<div style="text-align:center; margin-bottom:1rem;">
  <div style="color:#55557a; font-size:.72rem; overflow:hidden;
       white-space:nowrap; text-overflow:ellipsis;" title="{r['name']}">
    {r['name'][:22]}
  </div>
  <div style="color:{c_r}; font-size:1.4rem; font-weight:700;
       font-family:'DM Mono',monospace;">{r['score']:.1f}</div>
  <span class="{pill}">
    {'✅ GOOD' if r['verdict']=='Good' else '❌ BAD'}
  </span>
</div>""", unsafe_allow_html=True)
                else:
                    import pandas as pd
                    df = pd.DataFrame(results).drop(columns=["path"])
                    st.dataframe(df, use_container_width=True, height=500)


# =============================================================================
#  TAB 4 — TRAIN MODEL
# =============================================================================
with t4:
    st.markdown('<div class="sec-pill">Generate dataset and train blur_model.h5</div>',
                unsafe_allow_html=True)

    c_cfg1, c_cfg2 = st.columns(2)
    with c_cfg1:
        t_source  = st.text_input("Source images folder", value=BASE_DIR)
        t_dataset = st.text_input("Dataset output folder", value=DATASET_DIR)
        t_model   = st.text_input("Model save path (.h5)", value=MODEL_SAVE_PATH)
    with c_cfg2:
        t_epochs  = st.slider("Total epochs",  5, 60, 25)
        t_batch   = st.selectbox("Batch size", [8, 16, 32, 64], index=2)
        t_skip    = st.checkbox("Skip dataset generation (re-use existing)", value=False)

    # Info panel
    st.markdown("""
<div style="background:#13132a; border:1px solid #252550; border-radius:12px;
     padding:1.2rem 1.5rem; margin:1rem 0;">
  <div style="color:#818cf8; font-weight:600; margin-bottom:.8rem;">Training overview</div>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:.6rem;">
    <div style="color:#55557a; font-size:.82rem;">
      <span style="color:#e0e0ff;">Phase 1</span> — Head only, backbone frozen<br/>
      Fast convergence (~⅓ of total epochs)
    </div>
    <div style="color:#55557a; font-size:.82rem;">
      <span style="color:#e0e0ff;">Phase 2</span> — Fine-tune top 40% of MobileNetV2<br/>
      Lower LR (2e-5), higher final accuracy
    </div>
    <div style="color:#55557a; font-size:.82rem; margin-top:.4rem;">
      <span style="color:#e0e0ff;">CPU estimate</span><br/>
      100 source images → ~2–4 hours
    </div>
    <div style="color:#55557a; font-size:.82rem; margin-top:.4rem;">
      <span style="color:#e0e0ff;">GPU estimate</span><br/>
      100 source images → ~15–30 minutes
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("📦  Step 1 — Generate dataset only", use_container_width=True):
            if not os.path.isdir(t_source):
                st.error(f"Folder not found: {t_source}")
            else:
                with st.spinner("Generating synthetic blur dataset…"):
                    try:
                        lp = generate_dataset(t_source, t_dataset)
                        st.success(f"✅ Dataset ready → {lp}")
                    except Exception as e:
                        st.error(f"❌ {e}")

    with col_btn2:
        if st.button("🚀  Step 2 — Train model", use_container_width=True, type="primary"):
            labels_path = os.path.join(t_dataset, "labels.csv")
            ok = True
            if not t_skip and not os.path.isdir(t_source):
                st.error(f"Source folder not found: {t_source}"); ok = False
            if not t_skip and not os.path.exists(labels_path):
                st.warning("Dataset not found — run Step 1 first."); ok = False

            if ok:
                note = st.info("⏳ Training started … watch your terminal for epoch progress.")
                try:
                    if not t_skip:
                        generate_dataset(t_source, t_dataset)
                    _, meta = train_model(
                        labels_path = os.path.join(t_dataset, "labels.csv"),
                        model_save  = t_model,
                        epochs      = t_epochs,
                        batch_size  = t_batch,
                    )
                    note.empty()
                    st.success("✅ Training complete!")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy",  f"{meta['test_accuracy_pct']:.1f}%")
                    m2.metric("F1 Score",  f"{meta['test_f1_pct']:.1f}%")
                    m3.metric("MAE",       f"{meta['test_mae']:.4f}")
                    m4.metric("Samples",   meta['total_samples'])
                    st.info("🔄 Copy the new blur_model.h5 to your Bulk_image folder, "
                            "then refresh this dashboard to use it.")
                except Exception as e:
                    note.empty()
                    st.error(f"❌ Training error: {e}")


# =============================================================================
#  TAB 5 — SCORE GUIDE
# =============================================================================
with t5:
    st.markdown('<div class="sec-pill">What each score means — and how it is calculated</div>',
                unsafe_allow_html=True)

    # ── Score reference ───────────────────────────────────────────────────────
    st.markdown("#### Quality Score Reference")
    for lo, hi, label, color in QUALITY_TABLE:
        is_g   = lo >= 55
        v_col  = "#10b981" if is_g else "#ef4444"
        v_text = "Good Quality" if is_g else "Bad Quality"
        st.markdown(f"""
<div style="display:flex; align-items:center; gap:1rem; margin-bottom:.5rem;
     background:#13132a; border-radius:10px; padding:.7rem 1rem;
     border:1px solid #1e1e35;">
  <div style="width:70px; text-align:center; font-family:'DM Mono',monospace;
       font-size:.88rem; font-weight:600; color:{color}; flex-shrink:0;">
    {lo}–{hi}
  </div>
  <div style="flex:1; background:#0d0d18; height:9px; border-radius:5px; overflow:hidden;">
    <div style="width:{hi}%; height:100%; background:linear-gradient(90deg,{color}60,{color});
         border-radius:5px;"></div>
  </div>
  <div style="width:140px; color:{color}; font-weight:600; flex-shrink:0;">{label}</div>
  <div style="width:110px; color:{v_col}; font-size:.82rem; flex-shrink:0;">
    {'✅' if is_g else '❌'} {v_text}
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── How scores are calculated ─────────────────────────────────────────────
    st.markdown("#### How the score is calculated")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("""
<div style="background:#13132a; border:1px solid #252550; border-radius:14px;
     padding:1.3rem; height:100%;">
  <div style="color:#818cf8; font-weight:700; margin-bottom:.8rem;">
    🏠 Floor Plan
  </div>
  <div style="color:#55557a; font-size:.82rem; line-height:1.8;">
    <b style="color:#e0e0ff;">Problem:</b><br/>
    White background (70%+ of pixels) fools standard blur detectors.<br/><br/>
    <b style="color:#e0e0ff;">Solution:</b><br/>
    Threshold → extract line mask only.<br/>
    Measure Laplacian only on line pixels.<br/>
    White areas are 100% ignored.<br/><br/>
    <span style="color:#818cf8;">
      Line sharpness ×0.70<br/>+ CV base ×0.30
    </span>
  </div>
</div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
<div style="background:#13132a; border:1px solid #252550; border-radius:14px;
     padding:1.3rem; height:100%;">
  <div style="color:#a78bfa; font-weight:700; margin-bottom:.8rem;">
    🗺️ Master Plan
  </div>
  <div style="color:#55557a; font-size:.82rem; line-height:1.8;">
    <b style="color:#e0e0ff;">Problem:</b><br/>
    Flat colour regions inflate variance scores artificially.<br/><br/>
    <b style="color:#e0e0ff;">Solution:</b><br/>
    Canny edge detection to find colour boundaries.<br/>
    Measure sharpness only at those boundaries.<br/><br/>
    <span style="color:#a78bfa;">
      Boundary sharpness ×0.65<br/>+ Tenengrad ×0.35
    </span>
  </div>
</div>""", unsafe_allow_html=True)

    with col_c:
        st.markdown("""
<div style="background:#13132a; border:1px solid #252550; border-radius:14px;
     padding:1.3rem; height:100%;">
  <div style="color:#10b981; font-weight:700; margin-bottom:.8rem;">
    🖼️ Gallery Photo
  </div>
  <div style="color:#55557a; font-size:.82rem; line-height:1.8;">
    <b style="color:#e0e0ff;">4 metrics blended:</b><br/>
    • Laplacian variance (35%)<br/>
    • Tenengrad gradient (30%)<br/>
    • FFT high-frequency energy (20%)<br/>
    • Block texture variance (15%)<br/><br/>
    <span style="color:#10b981;">
      Hard cap applied if Laplacian<br/>falls below critical thresholds.
    </span>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── AI + CV blend explanation ─────────────────────────────────────────────
    st.markdown("#### AI model + CV algorithm blend")
    st.markdown("""
<div style="background:#13132a; border:1px solid #252550; border-radius:14px;
     padding:1.3rem;">
  <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem;
       font-size:.82rem; color:#55557a; line-height:1.8;">
    <div>
      <span style="color:#818cf8; font-weight:600;">Without model (CV only)</span><br/>
      Runs 100% on OpenCV metrics.<br/>
      Works immediately, no training required.<br/>
      Good enough for most use cases.
    </div>
    <div>
      <span style="color:#10b981; font-weight:600;">With model (AI + CV blend)</span><br/>
      AI score × blend weight<br/>
      + CV score × (1 − weight)<br/>
      Default weight = 0.60 (60% AI, 40% CV).
    </div>
    <div>
      <span style="color:#f59e0b; font-weight:600;">Hard caps always apply</span><br/>
      Even with a high AI score, if Laplacian<br/>
      variance or contrast is critically low,<br/>
      the score is capped to protect against<br/>
      false positives on truly blank images.
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Integration snippet ───────────────────────────────────────────────────
    st.markdown("#### How to use in app.py (3-line change)")
    st.code("""
# 1. Add import at top of app.py
from train import assess_image_quality

# 2. Add to CONFIG section
BLUR_MODEL_PATH = os.path.join(BASE_DIR, "blur_model.h5")

# 3. Inside process_single_image() — replace the old call with:
quality_score, metrics = assess_image_quality(
    image_path,
    category   = category,          # "floorplan" | "masterplan" | "gallery"
    model_path = BLUR_MODEL_PATH,   # omit or set None to use CV-only mode
)
""", language="python")


# =============================================================================
#  FOOTER
# =============================================================================
st.markdown("""
<div style="text-align:center; padding:2rem; margin-top:3rem;
     background:linear-gradient(135deg,#6366f110,#10b98108);
     border-radius:16px; border:1px solid #1e1e35;">
  <div style="font-size:1.3rem; font-weight:700;
       background:linear-gradient(135deg,#818cf8,#10b981);
       -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    🏠 HOMES247 — Blur Detection System
  </div>
  <div style="color:#33334a; font-size:.82rem; margin-top:.4rem;">
    train.py + blur_dashboard.py · Production ready
  </div>
</div>
""", unsafe_allow_html=True)