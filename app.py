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
)

VERDICT_STYLE = {
    "AUTHENTIC": {"color": "#1B8A3E", "bg": "#E6F7EC", "label": "AUTENTIKE"},
    "SUSPICIOUS": {"color": "#B8860B", "bg": "#FFF7E0", "label": "E DYSHIMTË"},
    "FAKE": {"color": "#C0392B", "bg": "#FDECEA", "label": "E MANIPULUAR"},
    "MANIPULATED": {"color": "#C0392B", "bg": "#FDECEA", "label": "E MANIPULUAR"},
}

DEFAULT_STYLE = {"color": "#555555", "bg": "#F0F0F0", "label": "PA KLASIFIKIM"}

# ---------------------------------------------------------------------------
# Header
# -------------------------------------------------------------------------
st.title("🔎 Sistemi i Analizës Forenzike të Provave Gjyqësore")
st.caption(
    "Vendos imazhin e prezantuar si provë gjyqësore për t'u analizuar "
    "për shenja manipulimi ose gjenerimi sintetik (deepfake)."
)

st.divider()

# ---------------------------------------------------------------------------
# Ngarkimi i imazhit
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Ngarko imazhin (provë gjyqësore)",
    type=["jpg", "jpeg", "png"],
    help="Formatet e pranuara: JPG, JPEG, PNG",
)

analyze_clicked = False

if uploaded_file is not None:
    col_preview, col_info = st.columns([1, 2])
    with col_preview:
        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
    with col_info:
        st.write(f"**Emri i skedarit:** {uploaded_file.name}")
        st.write(f"**Madhësia:** {uploaded_file.size / 1024:.1f} KB")
        st.write(f"**Data e ngarkimit:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        analyze_clicked = st.button("🔬 Fillo Analizën Forenzike", type="primary")

st.divider()

# ---------------------------------------------------------------------------
# Analiza
# ---------------------------------------------------------------------------
if analyze_clicked and uploaded_file is not None:
    with st.spinner("Duke ekzekutuar analizën forenzike (integritet → CNN → forensic → fusion → explainability)..."):
        # Ruaj skedarin e ngarkuar përkohësisht, sepse pipeline-i pret një path
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

# ---------------------------------------------------------------------------
# Rezultatet
# ---------------------------------------------------------------------------
if "last_result" in st.session_state:
    result = st.session_state["last_result"]

    decision = result["fusion"]["decision"]
    classification = decision["classification"]
    confidence = decision["confidence"]
    prob_fake = result["fusion"]["probability_fake"]

    style = VERDICT_STYLE.get(classification, DEFAULT_STYLE)

    st.subheader("📋 Rezultati i Analizës")

    st.markdown(
        f"""
        <div style="background-color:{style['bg']}; padding: 24px; border-radius: 10px;
                    border-left: 8px solid {style['color']}; margin-bottom: 20px;">
            <span style="font-size: 14px; color: #555;">VERDIKTI</span><br>
            <span style="font-size: 32px; font-weight: 700; color: {style['color']};">
                {style['label']}
            </span>
            <span style="font-size: 16px; color: #555; margin-left: 12px;">
                (besueshmëri: {confidence})
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Probabiliteti P(fake)", f"{prob_fake:.1%}")
    col_b.metric("ID e Provës", result["evidence_id"])
    col_c.metric("Rregulla Vendimi", decision["decision_rule"])

    st.divider()

    # -- Scores breakdown ----------------------------------------------------
    st.subheader("📊 Ndarja e Sinjaleve (Scores)")
    scores = result["scores"]
    score_cols = st.columns(len(scores))
    score_labels = {"cnn": "CNN", "ela": "ELA", "cfa": "CFA", "dct": "DCT", "fft": "FFT"}
    for col, (key, value) in zip(score_cols, scores.items()):
        label = score_labels.get(key) or key.upper()
        col.metric(label, f"{value:.4f}")

    st.caption(
        "Shënim: sinjalet ELA/CFA/DCT/FFT janë forenzike tradicionale/frekuence; "
        "kontributi kryesor i klasifikimit vjen nga CNN (shih 'Kufizimet' më poshtë)."
    )

    st.divider()

    # -- Explainability panels ------------------------------------------------
    st.subheader("🖼️ Paneli i Shpjegueshmërisë (Explainability)")
    panels = result["explainability"]["panel_paths"]
    panel_labels = {
        "original": "Origjinali",
        "ela": "ELA Heatmap",
        "gradcam": "Grad-CAM (CNN)",
        "overlay": "Overlay i Kombinuar",
    }

    panel_cols = st.columns(len(panels))
    for col, (key, path) in zip(panel_cols, panels.items()):
        img_path = Path(path)
        caption = panel_labels.get(key) or key
        if img_path.exists():
            col.image(str(img_path), caption=caption, use_container_width=True)
        else:
            col.warning(f"Panel '{key}' s'u gjet te {path}")

    iou = result["explainability"].get("ela_gradcam_agreement_iou")
    if iou is not None:
        st.caption(f"Përputhja hapësinore ELA↔Grad-CAM (IoU): {iou:.3f}")

    st.divider()

    # -- Integrity -------------------------------------------------------------
    with st.expander("🔒 Integriteti dhe Verifikimi i Skedarit"):
        integrity = result["integrity"]
        st.write(f"**SHA-256:** `{integrity['sha256']}`")
        fv = integrity["file_validation"]
        st.write(f"**Ekstensioni:** {fv['extension']}")
        st.write(f"**MIME i pritur:** {fv['expected_mime']}")
        st.write(f"**MIME real (magic bytes):** {fv['actual_mime_from_magic_bytes']}")
        st.write(f"**Konsistent:** {'✅ Po' if fv['is_consistent'] else '⚠️ Jo'}")

    # -- Chain of custody --------------------------------------------------
    with st.expander("⛓️ Zinxhiri i Kujdestarisë (Chain of Custody)"):
        for event in result["chain_of_custody"]:
            st.write(f"**{event['step']}** — {event['timestamp']}")
            if event.get("output_summary"):
                st.json(event["output_summary"])

    # -- Limitations ---------------------------------------------------------
    with st.expander("⚠️ Kufizimet e Sistemit", expanded=True):
        for limitation in result["limitations"]:
            st.markdown(f"- {limitation}")

    st.info(result["disclaimer_text"])

    st.divider()

    # -- Downloads -------------------------------------------------------------
    st.subheader("⬇️ Shkarkimet")

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

else:
    st.info("Ngarko një imazh më sipër dhe kliko 'Fillo Analizën Forenzike' për të parë rezultatet.")