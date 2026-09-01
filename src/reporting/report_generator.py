"""
report_generator.py — Krijon raport PDF forenzik profesional, i
strukturuar sipas: Case Summary -> A.Evidence -> B.ML -> C.Fusion ->
D.Explainability -> E.Conclusion -> Limitations -> Chain of Custody.
Ekzekutim: python src\reporting\report_generator.py <path_to_inference_json>
"""
import sys
import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
)

OUTPUT_DIR = Path("outputs/reports")

CUSTODY_STEP_LABELS = {
    "evidence_received": "Evidence received",
    "integrity_validation": "Integrity verification",
    "preprocessing": "Preprocessing",
    "cnn_analysis": "CNN analysis",
    "forensic_analysis_started": "Forensic analysis started",
    "forensic_analysis_completed": "Forensic analysis completed",
    "fusion_classification": "Fusion classification",
    "explainability_generated": "Explainability generated",
    "report_data_finalized": "Report generated",
}


def build_styles():
    styles = getSampleStyleSheet()

    # Mbishkruaj stilet bazë me Times family + leading eksplicit
    styles["Normal"].fontName = "Times-Roman"
    styles["Normal"].fontSize = 10
    styles["Normal"].leading = 13

    styles.add(ParagraphStyle(
        name="ReportTitle", fontName="Times-Bold", fontSize=18,
        leading=22, spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle", fontName="Times-Roman", fontSize=9,
        leading=11, textColor=colors.grey, spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        name="SectionHeader", fontName="Times-Bold", fontSize=13,
        leading=16, spaceBefore=14, spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name="SubHeader", fontName="Times-Bold", fontSize=11,
        leading=14, spaceBefore=10, spaceAfter=4,
        textColor=colors.HexColor("#333333")
    ))
    styles.add(ParagraphStyle(
        name="Disclaimer", fontName="Times-Italic", fontSize=8,
        leading=11, textColor=colors.HexColor("#555555"),
        spaceAfter=6, spaceBefore=6
    ))
    styles.add(ParagraphStyle(
        name="MonoSmall", fontName="Courier", fontSize=8, leading=11
    ))
    styles.add(ParagraphStyle(
        name="TableKey", fontName="Times-Bold", fontSize=9, leading=12
    ))
    styles.add(ParagraphStyle(
        name="TableValue", fontName="Times-Roman", fontSize=9, leading=12
    ))
    styles.add(ParagraphStyle(
        name="ClassResultBig", fontName="Times-Bold", fontSize=28,
        leading=34, spaceAfter=4
        # ngjyra i shtohet dinamikisht te generate_report()
    ))

    return styles


def kv_table(pairs: list[tuple[str, str]], styles, col_widths=(5.5 * cm, 10.5 * cm)) -> Table:
    mono_style = styles["MonoSmall"]
    value_style = styles["TableValue"]
    key_style = styles["TableKey"]

    data = [
        [
            Paragraph(str(k), key_style),
            Paragraph(str(v), mono_style if ("SHA" in k or "ID" in k) else value_style),
        ]
        for k, v in pairs
    ]

    table = Table(data, colWidths=list(col_widths))
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f2f2f2")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def generate_report(inference_json_path: str) -> str:
    with open(inference_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    styles = build_styles()
    story = []

    evidence_id = data["evidence_id"]
    out_path = OUTPUT_DIR / f"{evidence_id}_report.pdf"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scores = data["scores"]
    fusion = data["fusion"]
    decision = fusion["decision"]

    # ============================================================
    # FAQJA 1 — PËRMBLEDHJE EKZEKUTIVE (verdikti i parë, i madh, qartë)
    # ============================================================
    story.append(Paragraph("Forensic Deepfake Analysis Report", styles["ReportTitle"]))
    story.append(Paragraph(f"Generated: {data['generated_at']}  |  Evidence ID: {evidence_id}",
                            styles["ReportSubtitle"]))
    story.append(Spacer(1, 14))

    classification_color = {
        "AUTHENTIC": colors.green, "SUSPICIOUS": colors.orange, "MANIPULATED": colors.red
    }.get(decision["classification"], colors.black)

    class_style = ParagraphStyle(name="ClassResultBig", fontSize=28,
                                  textColor=classification_color, fontName="Helvetica-Bold",
                                  spaceAfter=4)
    story.append(Paragraph(decision["classification"], class_style))
    story.append(Spacer(1, 3))
    story.append(Paragraph(f"Decision confidence: {decision['confidence']}", styles["SubHeader"]))
    story.append(Spacer(1, 10))

    story.append(kv_table([
        ("Fusion probability P(fake)", f"{fusion['probability_fake']:.4f}"),
        ("Decision rule applied", decision["decision_rule"]),
        ("Original filename", data["original_filename"]),
        ("File size (bytes)", data["file_size_bytes"]),
        ("SHA-256 (evidence hash)", data["integrity"]["sha256"][:16] + "..."),
    ],styles))
    story.append(Spacer(1, 12))

    story.append(Paragraph(data["scope_text"], styles["Normal"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(data["disclaimer_text"], styles["Disclaimer"]))

    story.append(PageBreak())

    # ============================================================
    # FAQJA 2 — BAZA E VENDIMIT (numrat kyç, në gjuhë të thjeshtë)
    # ============================================================
    story.append(Paragraph("Basis of Conclusion", styles["SectionHeader"]))
    story.append(Paragraph(
        "The primary basis for this classification is the deep-learning (CNN) analysis, "
        "trained on over 40,000 labeled images. Traditional forensic signals (CFA, DCT, FFT) "
        "contribute a supporting role in the fusion decision; ELA is included for methodological "
        "transparency but does not independently discriminate authentic from manipulated content "
        "in this system (see Limitations).",
        styles["Normal"]
    ))
    story.append(Spacer(1, 10))

    story.append(kv_table([
        ("CNN probability (fake) — primary signal", f"{scores['cnn']:.4f}"),
        ("Fusion probability P(fake) — final combined score", f"{fusion['probability_fake']:.4f}"),
        ("Threshold T1 (Authentic, high confidence, below)", f"{fusion['t1_real_threshold']:.4f}"),
        ("Threshold T2 (Manipulated, high confidence, above)", f"{fusion['t2_fake_threshold']:.4f}"),
    ],styles))
    story.append(Spacer(1, 10))

    # --- Explainability e sjellë KËTU, jo si seksion D i ndarë ---
    story.append(Paragraph("Visual Evidence (CNN Attention & ELA)", styles["SubHeader"]))
    panels = data["explainability"]["panel_paths"]
    img_w, img_h = 6 * cm, 6 * cm
    grid_data = [
        [Paragraph("<b>Original Image</b>"), Paragraph("<b>ELA Heatmap</b>")],
        [RLImage(panels["original"], width=img_w, height=img_h),
         RLImage(panels["ela"], width=img_w, height=img_h)],
        [Paragraph("<b>Grad-CAM (CNN Attention)</b>"), Paragraph("<b>Evidence Overlay</b>")],
        [RLImage(panels["gradcam"], width=img_w, height=img_h),
         RLImage(panels["overlay"], width=img_w, height=img_h)],
    ]
    grid_table = Table(grid_data, colWidths=[img_w + 0.3 * cm, img_w + 0.3 * cm])
    grid_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(grid_table)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"ELA–Grad-CAM spatial agreement (IoU): {data['explainability']['ela_gradcam_agreement_iou']:.4f} "
        f"(see Limitations for interpretation)",
        ParagraphStyle(name="SmallNote", fontSize=8, textColor=colors.grey)
    ))

    story.append(PageBreak())

    # ============================================================
    # ANEKS TEKNIK — A. Evidence Examination
    # ============================================================
    story.append(Paragraph("Technical Appendix", styles["ReportTitle"]))
    story.append(Paragraph(
        "The following sections provide full technical detail for expert review. "
        "They are not required reading to understand the conclusion above.",
        styles["Disclaimer"]
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("A. Evidence Examination", styles["SectionHeader"]))
    story.append(Paragraph("A.1 File Integrity", styles["SubHeader"]))
    validation = data["integrity"]["file_validation"]
    story.append(kv_table([
        ("Extension", validation["extension"]),
        ("Expected MIME type", validation["expected_mime"]),
        ("Actual MIME (magic bytes)", validation["actual_mime_from_magic_bytes"]),
        ("Consistency", "PASSED" if validation["is_consistent"] else "FLAGGED — MISMATCH"),
    ],styles))

    story.append(Paragraph("A.2 Forensic Signal Scores (supporting signals, not standalone verdicts)",
                            styles["SubHeader"]))
    story.append(kv_table([
        ("ELA score", f"{scores['ela']:.4f}  (no independent discriminative signal)"),
        ("CFA score", f"{scores['cfa']:.4f}  (weak supporting signal)"),
        ("DCT score", f"{scores['dct']:.4f}  (weak supporting signal)"),
        ("FFT score", f"{scores['fft']:.4f}  (weak supporting signal)"),
    ],styles))
    story.append(Spacer(1, 8))

    # B. Machine-Learning Examination
    story.append(Paragraph("B. Machine-Learning Examination", styles["SectionHeader"]))
    cnn_meta = data["model_metadata"]["cnn"]
    story.append(kv_table([
        ("Architecture", cnn_meta["architecture"]),
        ("Model version", cnn_meta["version"]),
        ("Framework", cnn_meta["framework"]),
        ("Input resolution", cnn_meta["input_resolution"]),
        ("Training datasets", ", ".join(cnn_meta["training_datasets"])),
        ("Model checkpoint", cnn_meta["checkpoint"]),
        ("CNN probability (fake)", f"{scores['cnn']:.4f}"),
    ],styles))
    story.append(Spacer(1, 8))

    # C. Evidence Fusion
    story.append(Paragraph("C. Evidence Fusion", styles["SectionHeader"]))
    fusion_meta = data["model_metadata"]["fusion"]
    story.append(kv_table([
        ("Fusion algorithm", fusion_meta["algorithm"]),
        ("Fusion version", fusion_meta["version"]),
        ("Features used", ", ".join(fusion["feature_names"])),
        ("P(fake)", f"{fusion['probability_fake']:.4f}"),
        ("Threshold T1", f"{fusion['t1_real_threshold']:.4f}"),
        ("Threshold T2", f"{fusion['t2_fake_threshold']:.4f}"),
    ],styles))
    story.append(Spacer(1, 8))

    # --- Limitations ---
    story.append(Paragraph("Limitations", styles["SubHeader"]))
    for limitation in data["limitations"]:
        story.append(Paragraph(f"• {limitation}", styles["Normal"]))
        story.append(Spacer(1, 3))
    story.append(Spacer(1, 8))

    # --- Chain of Custody ---
    story.append(Paragraph("Chain of Custody", styles["SubHeader"]))
    custody_log = data.get("chain_of_custody", [])
    for entry in custody_log:
        label = CUSTODY_STEP_LABELS.get(entry["step"], entry["step"])
        timestamp = entry["timestamp"][:19].replace("T", " ")
        version = entry.get("model_version")
        version_str = f" (model: {version})" if version else ""
        story.append(Paragraph(f"{label}: {timestamp}{version_str}", styles["Normal"]))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 6))
    story.append(Paragraph("Operator/System: Deepfake Detection System v1.0", styles["Normal"]))

    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                             topMargin=2 * cm, bottomMargin=2 * cm,
                             leftMargin=2 * cm, rightMargin=2 * cm)
    doc.build(story)

    return str(out_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Përdorim: python report_generator.py <path_to_inference_json>")
        sys.exit(1)

    report_path = generate_report(sys.argv[1])
    print(f"Raporti u gjenerua te: {report_path}")