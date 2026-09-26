"""
app/pdf_report.py
=================
Clinical PDF Screening Report Generator for Diabetic Retinopathy Triage.

Generates an audit-ready, professional medical consultation report modeled
after diagnostic screening platforms (EyeArt, IDx-DR). Incorporates side-by-side
retinal imaging, Grad-CAM heatmaps, quantitative confidence breakdown,
triage recommendations, and physician sign-off blocks.

Author: Sanduni Herath
Repository: https://github.com/SanduniHerath/diabetic-retinopathy-cv-assignment
"""

import io
from datetime import datetime
from typing import Any, Dict

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def generate_clinical_pdf_report(result: Dict[str, Any]) -> bytes:
    """
    Generates a structured clinical PDF report in memory.

    Args:
        result: Dictionary returned by predictor.predict_image()

    Returns:
        Binary bytes of the compiled PDF document.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.50 * inch,
        bottomMargin=0.50 * inch,
    )

    styles = getSampleStyleSheet()

    # Custom Medical Styling Hierarchy
    header_style = ParagraphStyle(
        "ClinHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0F172A"),  # Slate-900
    )
    sub_header_style = ParagraphStyle(
        "ClinSubHeader",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#475569"),  # Slate-600
    )
    section_title_style = ParagraphStyle(
        "ClinSectionTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#1E3A8A"),  # Blue-900
        spaceBefore=5,
        spaceAfter=3,
    )
    body_style = ParagraphStyle(
        "ClinBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1E293B"),
    )
    bullet_style = ParagraphStyle(
        "ClinBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#334155"),
        leftIndent=10,
    )
    table_label_style = ParagraphStyle(
        "ClinTableLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#475569"),
    )
    table_val_style = ParagraphStyle(
        "ClinTableVal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0F172A"),
    )
    disclaimer_style = ParagraphStyle(
        "ClinDisclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=6.8,
        leading=8.5,
        textColor=colors.HexColor("#64748B"),
    )

    elements = []

    # -----------------------------------------------------------------------
    # 1. Header & Clinic Branding
    # -----------------------------------------------------------------------
    exam_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exam_id = f"EXAM-{datetime.now().strftime('%Y%m%d')}-{random_suffix()}"

    header_data = [
        [
            Paragraph("<b>RETINASYSTEM AI — CLINICAL SCREENING REPORT</b>", header_style),
            Paragraph(f"<b>Exam ID:</b> {exam_id}<br/><b>Date:</b> {exam_timestamp}", sub_header_style),
        ],
        [
            Paragraph("Automated Diabetic Retinopathy Tele-screening & Severity Staging Decision Support System", sub_header_style),
            Paragraph("<b>Regulatory Class:</b> Investigational SaMD", sub_header_style),
        ],
    ]
    header_table = Table(header_data, colWidths=[4.7 * inch, 2.3 * inch])
    header_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
        ])
    )
    elements.append(header_table)
    elements.append(Spacer(1, 4))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceBefore=2, spaceAfter=6))

    # -----------------------------------------------------------------------
    # 2. Patient & Examination Metadata Table
    # -----------------------------------------------------------------------
    patient_id = result.get("patient_id", "PT-8291")
    eye = result.get("eye", "OD (Right Eye)")
    engine = result.get("engine_mode", "EfficientNet-B0 Dual-Curriculum")

    meta_data = [
        [
            Paragraph("<b>Patient ID:</b>", table_label_style),
            Paragraph(patient_id, table_val_style),
            Paragraph("<b>Examined Eye:</b>", table_label_style),
            Paragraph(eye, table_val_style),
        ],
        [
            Paragraph("<b>Referring Clinic:</b>", table_label_style),
            Paragraph("Endocrinology & Diabetic Eye Clinic", table_val_style),
            Paragraph("<b>Analysis Engine:</b>", table_label_style),
            Paragraph(engine, table_val_style),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[1.3 * inch, 2.2 * inch, 1.3 * inch, 2.2 * inch])
    meta_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ])
    )
    elements.append(meta_table)
    elements.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # 3. Primary Finding & Triage Badge
    # -----------------------------------------------------------------------
    details = result["details"]
    triage_cat = result["triage_category"]

    if triage_cat == "urgent":
        badge_bg = colors.HexColor("#FEF2F2")
        badge_border = colors.HexColor("#EF4444")
        badge_text_col = colors.HexColor("#991B1B")
    elif triage_cat == "routine":
        badge_bg = colors.HexColor("#FFFBEB")
        badge_border = colors.HexColor("#F59E0B")
        badge_text_col = colors.HexColor("#92400E")
    else:
        badge_bg = colors.HexColor("#ECFDF5")
        badge_border = colors.HexColor("#10B981")
        badge_text_col = colors.HexColor("#065F46")

    diag_left = [
        Paragraph(f"<b>Primary Finding:</b> <font size=12 color='#0F172A'><b>{details['full_name']}</b></font>", body_style),
        Paragraph(f"<b>Clinical Severity Grade:</b> {details['grade']} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Model Confidence:</b> <b>{result['confidence_pct']}</b>", body_style),
    ]

    triage_cell = [
        Paragraph("<b>CLINICAL TRIAGE STATUS:</b>", ParagraphStyle("TLabel", fontName="Helvetica-Bold", fontSize=7.5, textColor=badge_text_col)),
        Paragraph(f"<b>{result['triage_label'].upper()}</b>", ParagraphStyle("TVal", fontName="Helvetica-Bold", fontSize=11, leading=13, textColor=badge_text_col)),
        Paragraph(f"Action Window: {result['triage_timeline']}", ParagraphStyle("TTime", fontName="Helvetica", fontSize=7.5, textColor=badge_text_col)),
    ]

    finding_table_data = [[diag_left, triage_cell]]
    finding_table = Table(finding_table_data, colWidths=[4.6 * inch, 2.4 * inch])
    finding_table.setStyle(
        TableStyle([
            ("BACKGROUND", (1, 0), (1, 0), badge_bg),
            ("BOX", (1, 0), (1, 0), 1.2, badge_border),
            ("BOX", (0, 0), (0, 0), 0.5, colors.HexColor("#CBD5E1")),
            ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#FFFFFF")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    elements.append(finding_table)
    elements.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # 4. Side-by-Side Retinal Inspection (Fundus Image + Grad-CAM Heatmap)
    # -----------------------------------------------------------------------
    elements.append(Paragraph("<b>Retinal Inspection & Grad-CAM Spatial Explainability</b>", section_title_style))

    def _optimize_img_for_pdf(raw_bytes: bytes, max_dim: int = 500) -> io.BytesIO:
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(raw_bytes)).convert("RGB")
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), PILImage.Resampling.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=85)
        out.seek(0)
        return out

    orig_buf = _optimize_img_for_pdf(result["original_image_bytes"])
    grad_buf = _optimize_img_for_pdf(result["gradcam_image_bytes"])

    img_width = 3.25 * inch
    img_height = 2.45 * inch

    img_left = RLImage(orig_buf, width=img_width, height=img_height)
    img_right = RLImage(grad_buf, width=img_width, height=img_height)

    img_captions = [
        Paragraph("<b>Figure A: Patient Fundus Photograph</b><br/>Preprocessed retinal field (224x224 input crop).", sub_header_style),
        Paragraph("<b>Figure B: Grad-CAM Activation Heatmap</b><br/>Warmer colours denote high diagnostic focus areas.", sub_header_style),
    ]

    image_table = Table([[img_left, img_right], img_captions], colWidths=[3.5 * inch, 3.5 * inch])
    image_table.setStyle(
        TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ])
    )
    elements.append(image_table)
    elements.append(Spacer(1, 5))

    # -----------------------------------------------------------------------
    # 5. Clinical Findings & Recommended Actions
    # -----------------------------------------------------------------------
    findings_list = [Paragraph(f"• {f}", bullet_style) for f in details["pathology_findings"]]
    actions_list = [Paragraph(f"• {a}", bullet_style) for a in details["recommendations"]]

    two_col_data = [
        [
            Paragraph("<b>Pathological Feature Assessment</b>", section_title_style),
            Paragraph("<b>Recommended Clinical Protocol</b>", section_title_style),
        ],
        [
            findings_list,
            actions_list,
        ],
    ]
    two_col_table = Table(two_col_data, colWidths=[3.5 * inch, 3.5 * inch])
    two_col_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 1), (0, 1), 0.5, colors.HexColor("#E2E8F0")),
            ("BOX", (1, 1), (1, 1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 1), (-1, 1), 5),
            ("RIGHTPADDING", (0, 1), (-1, 1), 5),
        ])
    )
    elements.append(two_col_table)
    elements.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # 6. Physician Sign-off & Verification Block
    # -----------------------------------------------------------------------
    sign_data = [
        [
            Paragraph("<b>Reviewing Clinician Name:</b> ___________________________", body_style),
            Paragraph("<b>Medical Registration No:</b> _______________", body_style),
            Paragraph("<b>Signature:</b> ______________________", body_style),
            Paragraph("<b>Date:</b> _________", body_style),
        ],
        [
            Paragraph("<b>Clinical Concurrence:</b> [  ] Approved as Classified &nbsp;&nbsp;&nbsp;&nbsp; [  ] Overridden / Reclassified to: _______________", body_style),
            "", "", "",
        ],
    ]
    sign_table = Table(sign_data, colWidths=[2.3 * inch, 1.8 * inch, 1.9 * inch, 1.0 * inch])
    sign_table.setStyle(
        TableStyle([
            ("SPAN", (0, 1), (-1, 1)),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ])
    )
    elements.append(KeepTogether([
        Paragraph("<b>Attending Ophthalmologist Review & Verification</b>", section_title_style),
        sign_table,
    ]))
    elements.append(Spacer(1, 4))

    # -----------------------------------------------------------------------
    # 7. Regulatory & Medical Disclaimer
    # -----------------------------------------------------------------------
    disclaimer_text = (
        "<b>CLINICAL DISCLAIMER & INVESTIGATIONAL USE NOTICE:</b> This report is generated by an artificial "
        "intelligence decision-support system trained on retinal fundus photography. It is designed to assist "
        "clinical triage workflows and is NOT a standalone diagnostic device. Image artefacts, media opacities "
        "(e.g. dense cataracts), or non-diabetic retinopathies may influence model saliency. Final diagnostic "
        "and therapeutic determinations must be verified by a licensed ophthalmologist through clinical examination."
    )
    disc_table = Table([[Paragraph(disclaimer_text, disclaimer_style)]], colWidths=[7.0 * inch])
    disc_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    elements.append(disc_table)

    # Build PDF document
    doc.build(elements)
    return buffer.getvalue()


def random_suffix() -> str:
    """Generates a pseudo-random clinical exam serial string."""
    import random
    return f"{random.randint(1000, 9999)}"
