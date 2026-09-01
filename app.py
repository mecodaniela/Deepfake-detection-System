"""
app.py — Front-end Streamlit për Sistemin e Detektimit të Deepfake-ve
(Analizë Forenzike për Prova Gjyqësore).
Ekzekutim: streamlit run app.py
"""
import sys
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.inference_pipeline import run_inference_pipeline
from src.reporting.report_generator import generate_report

# ---------------------------------------------------------------------------
# Konfigurimi i faqes
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Sistemi i Analizës Forenzike të Provave Gjyqësore",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS i personalizuar
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .hero {
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #1B4F72 0%, #2E86AB 100%);
        color: white;
        margin-bottom: 1.8rem;
    }
    .hero h1 {
        font-size: 2.0rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        color: white;
    }
    .hero p {
        font-size: 1.02rem;
        opacity: 0.92;
        margin-bottom: 0;
    }

    .info-card {
        background-color: #F4F6F8;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        border: 1px solid #E3E7EA;
    }

    .verdict-card {
        padding: 1.6rem 1.8rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
    }
    .verdict-label {
        font-size: 0.82rem;
        letter-spacing: 0.06em;
        font-weight: 600;
        opacity: 0.75;
        text-transform: uppercase;
    }
    .verdict-value {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0.15rem 0 0.3rem 0;
    }
    .verdict-sub {
        font-size: 0.95rem;
        opacity: 0.85;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 1.6rem 0 0.6rem 0;
        color: #1B4F72;
        border-bottom: 2px solid #E3E7EA;
        padding-bottom: 0.4rem;
    }

    div[data-testid="stMetric"] {
        background-color: #F4F6F8;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        border: 1px solid #E3E7EA;
    }

    div.stButton > button[kind="primary"] {
        background-color: #1B4F72;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.4rem;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #143C57;
    }

    .disclaimer-box {
        background-color: #FFF7E0;
        border-left: 4px solid #B8860B;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        font-size: 0.9rem;
        color: #5C4A0A;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

VERDICT_STYLE = {
    "AUTHENTIC": {"color": "#1B8A3E", "bg": "#E6F7EC", "label": "AUTENTIKE", "icon": "✅"},
    "SUSPICIOUS": {"color": "#B8860B", "bg": "#FFF7E0", "label": "E DYSHIMTË", "icon": "⚠️"},
    "FAKE": {"color": "#C0392B", "bg": "#FDECEA", "label": "E MANIPULUAR", "icon": "🚫"},
    "MANIPULATED": {"color": "#C0392B", "bg": "#FDECEA", "label": "E MANIPULUAR", "icon": "🚫"},
}
DEFAULT_STYLE = {"color": "#555555", "bg": "#F0F0F0", "label": "PA KLASIFIKIM", "icon": "❔"}

SCORE_LABELS = {"cnn": "CNN", "ela": "ELA", "cfa": "CFA", "dct": "DCT", "fft": "FFT"}
PANEL_LABELS = {
    "original": "Origjinali",
    "ela": "ELA Heatmap",
    "gradcam": "Grad-CAM (CNN)",
    "overlay": "Overlay i Kombinuar",
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🔎 Rreth Sistemit")
    st.markdown(
        "Sistem hibrid (CNN + forenzikë dixhitale + frekuencë) për zbulimin "
        "e manipulimeve deepfake në imazhe të paraqitura si provë gjyqësore."
    )
    st.divider()
    st.markdown("**Metodologjia**")
    st.markdown(
        "- CNN (EfficientNet-B0)\n"
        "- ELA, CFA, DCT (forenzikë)\n"
        "- FFT (frekuencë)\n"
        "- Fusion (Logistic Regression)\n"
        "- Explainability (Grad-CAM)"
    )
    st.divider()
    st.markdown(
        '<div class="disclaimer-box">⚠️ Output-i i sistemit është një vlerësim '
        'teknik probabilistik, jo provë ligjore vetvetiu.</div>',
        unsafe_allow_html=True,
    )
    st.divider()

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🔎 Sistemi i Analizës Forenzike të Provave Gjyqësore</h1>
        <p>Vendos një imazh të prezantuar si provë gjyqësore për t'u analizuar
        për shenja manipulimi ose gjenerimi sintetik (deepfake).</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_analyze, tab_about = st.tabs(["📤 Ngarko & Analizo", "ℹ️ Rreth Metodologjisë"])

with tab_analyze:
    uploaded_file = st.file_uploader(
        "Ngarko imazhin (provë gjyqësore)",
        type=["jpg", "jpeg", "png"],
        help="Formatet e pranuara: JPG, JPEG, PNG",
    )

    analyze_clicked = False

    if uploaded_file is not None:
        col_preview, col_info = st.columns([1, 1.4])
        with col_preview:
            st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
        with col_info:
            st.markdown(
                f"""
                <div class="info-card">
                    <b>Emri i skedarit:</b> {uploaded_file.name}<br>
                    <b>Madhësia:</b> {uploaded_file.size / 1024:.1f} KB<br>
                    <b>Data e ngarkimit:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            analyze_clicked = st.button("🔬 Fillo Analizën Forenzike", type="primary")

    if analyze_clicked and uploaded_file is not None:
        with st.spinner("Duke ekzekutuar analizën (integritet → CNN → forensic → fusion → explainability)..."):
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="outputs" if Path("outputs").exists() else None) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            try:
                result = run_inference_pipeline(tmp_path, output_dir="outputs/streamlit_inference")
            except Exception as e:
                st.error(f"Analiza dështoi: {e}")
                st.stop()

        st.session_state["last_result"] = result
        st.session_state["last_image_name"] = uploaded_file.name

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        decision = result["fusion"]["decision"]
        classification = decision["classification"]
        confidence = decision["confidence"]
        prob_fake = result["fusion"]["probability_fake"]
        style = VERDICT_STYLE.get(classification, DEFAULT_STYLE)

        st.markdown(
            f"""
            <div class="verdict-card" style="background-color:{style['bg']}; border-left: 8px solid {style['color']};">
                <div class="verdict-label">VERDIKTI</div>
                <div class="verdict-value" style="color:{style['color']};">
                    {style['icon']} {style['label']}
                </div>
                <div class="verdict-sub">Besueshmëri: {confidence}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Probabiliteti P(fake)", f"{prob_fake:.1%}")
        col_b.metric("ID e Provës", result["evidence_id"])
        col_c.metric("Rregulla Vendimi", decision["decision_rule"])

        st.markdown('<div class="section-title">📊 Ndarja e Sinjaleve (Scores)</div>', unsafe_allow_html=True)
        scores = result["scores"]
        score_cols = st.columns(len(scores))
        for col, (key, value) in zip(score_cols, scores.items()):
            label = SCORE_LABELS.get(key) or key.upper()
            col.metric(label, f"{value:.4f}")
        st.caption(
            "Sinjalet ELA/CFA/DCT/FFT janë forenzike tradicionale/frekuence; "
            "kontributi kryesor i klasifikimit vjen nga CNN."
        )

        st.markdown('<div class="section-title">🖼️ Paneli i Shpjegueshmërisë</div>', unsafe_allow_html=True)
        panels = result["explainability"]["panel_paths"]
        panel_cols = st.columns(len(panels))
        for col, (key, path) in zip(panel_cols, panels.items()):
            img_path = Path(path)
            caption = PANEL_LABELS.get(key) or key
            if img_path.exists():
                col.image(str(img_path), caption=caption, use_container_width=True)
            else:
                col.warning(f"Panel '{key}' s'u gjet te {path}")

        iou = result["explainability"].get("ela_gradcam_agreement_iou")
        if iou is not None:
            st.caption(f"Përputhja hapësinore ELA↔Grad-CAM (IoU): {iou:.3f}")

        with st.expander("🔒 Integriteti dhe Verifikimi i Skedarit"):
            integrity = result["integrity"]
            st.write(f"**SHA-256:** `{integrity['sha256']}`")
            fv = integrity["file_validation"]
            st.write(f"**Ekstensioni:** {fv['extension']}")
            st.write(f"**MIME i pritur:** {fv['expected_mime']}")
            st.write(f"**MIME real (magic bytes):** {fv['actual_mime_from_magic_bytes']}")
            st.write(f"**Konsistent:** {'✅ Po' if fv['is_consistent'] else '⚠️ Jo'}")

        with st.expander("⛓️ Chain of Custody"):
            for event in result["chain_of_custody"]:
                st.write(f"**{event['step']}** — {event['timestamp']}")
                if event.get("output_summary"):
                    st.json(event["output_summary"])

        with st.expander("⚠️ Kufizimet e Sistemit", expanded=True):
            for limitation in result["limitations"]:
                st.markdown(f"- {limitation}")

        st.markdown(f'<div class="disclaimer-box">{result["disclaimer_text"]}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">⬇️ Shkarkimet</div>', unsafe_allow_html=True)
        col_dl1, col_dl2 = st.columns(2)

        json_path = Path(result["_json_path"])
        if json_path.exists():
            with open(json_path, "rb") as f:
                col_dl1.download_button(
                    "📄 Shkarko Raportin JSON",
                    data=f,
                    file_name=f"{result['evidence_id']}_report.json",
                    mime="application/json",
                )

        if col_dl2.button("📑 Gjenero Raportin PDF"):
            with st.spinner("Duke gjeneruar PDF..."):
                try:
                    pdf_path = generate_report(str(json_path))
                    with open(pdf_path, "rb") as f:
                        col_dl2.download_button(
                            "📑 Shkarko Raportin PDF",
                            data=f,
                            file_name=f"{result['evidence_id']}_report.pdf",
                            mime="application/pdf",
                        )
                except Exception as e:
                    st.error(f"Gjenerimi i PDF-së dështoi: {e}")

    elif uploaded_file is None:
        st.info("Ngarko një imazh më sipër për të filluar analizën forenzike.")

with tab_about:
    st.markdown('<div class="section-title">🧬 Arkitektura e Sistemit</div>', unsafe_allow_html=True)
    st.markdown(
        """
        ```
                        ┌── CNN (EfficientNet-B0)
        Imazhi hyrës ───┼── Forenzikë (ELA + CFA + DCT)
                        └── Frekuencë (FFT)
                                │
                        Fusion (Logistic Regression)
                                │
                            P(fake) ∈ [0,1]
                                │
                    Vendim me prag të dyfishtë (T1/T2)
                                │
                Autentike / E dyshimtë / E manipuluar
        ```
        """
    )
    st.markdown('<div class="section-title">📈 Rezultatet Kryesore (test set, n=1000)</div>', unsafe_allow_html=True)
    st.table(
        {
            "Komponenti": ["CNN (frame-level)", "CNN (video-level)", "Fusion i plotë"],
            "Accuracy": ["85.19%", "90.00%", "90.10%"],
            "F1": ["0.8591", "0.9038", "0.8989"],
            "ROC-AUC": ["0.9235", "—", "0.9617"],
        }
    )
    st.markdown('<div class="section-title">⚠️ Kufizimet e Njohura</div>', unsafe_allow_html=True)
    st.markdown(
        "- Kalibruar për **face-swap/reenactment** (FaceForensics++, DFDC, Celeb-DF), jo për editues "
        "modernë diffusion-based (Gemini, DALL-E, Stable Diffusion)\n"
        "- Gjeneralizimi ndër-dataset bie nga ~90% në ~83% mbi dataset të panjohur\n"
        "- Accuracy përkeqësohet nën kompresim të fortë JPEG/humbje rezolucioni\n"
    )