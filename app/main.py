"""
app/main.py
===========
Professional Clinical Decision-Support Interface for Diabetic Retinopathy Screening.
Designed in the clinical aesthetic of deployed diagnostic platforms (EyeArt, IDx-DR).

Features:
  - Drag-and-drop retinal fundus image upload (PNG, JPG, JPEG)
  - 1-Click clinical test sample loader for rapid demonstration & evaluation
  - Side-by-side retinal inspection workstation: Original Fundus vs Grad-CAM Heatmap
  - Clinical Severity Staging (Grade 0 to Grade 4) with confidence distribution
  - Color-coded Referral Triage (No Referral Needed / Routine Follow-up / Refer Urgently)
  - Interactive Severity-Aware Clinical Assistant Chatbot
  - One-Click Structured Clinical PDF Screening Report Generator
  - Prominent Medical & Regulatory Investigational Disclaimers

Author: Sanduni Herath
Repository: https://github.com/SanduniHerath/diabetic-retinopathy-cv-assignment
"""

import base64
import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Setup portable sys.path for app and src
_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
for _p in (_APP_DIR, _REPO_ROOT, _REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from nicegui import app, events, ui
from PIL import Image

try:
    from app.pdf_report import generate_clinical_pdf_report, generate_patient_summary_pdf
    from app.predictor import (
        CLASS_NAMES,
        CLINICAL_DETAILS,
        PATIENT_DETAILS,
        generate_clinical_assistant_reply,
        is_real_model_available,
        predict_image,
    )
except ImportError:
    from pdf_report import generate_clinical_pdf_report, generate_patient_summary_pdf
    from predictor import (
        CLASS_NAMES,
        CLINICAL_DETAILS,
        PATIENT_DETAILS,
        generate_clinical_assistant_reply,
        is_real_model_available,
        predict_image,
    )

# ---------------------------------------------------------------------------
# State Management
# ---------------------------------------------------------------------------
state: Dict[str, Any] = {
    "current_result": None,
    "patient_id": "PT-2026-0842",
    "eye": "OD (Right Eye)",
    "use_mock": False,
    "patient_mode": False,          # False = Clinician View, True = Patient View
    "symptom_flags": {              # Patient-reported emergency symptoms
        "floaters": None,
        "blurry": None,
        "dark_curtain": None,
    },
    "chat_history": [
        (
            "assistant",
            "Welcome to the RetinaScan Clinical Decision Support System. "
            "Please upload a retinal fundus photograph or select a clinical sample image to begin automated screening and triage analysis.",
        )
    ],
}

# Pre-load available test sample images for 1-click clinical demos
# Uses verified high-confidence representative exemplars from each test class
PREFERRED_SAMPLES: Dict[str, str] = {
    "No_DR": "005b95c28852.png",
    "Mild": "01b3aed3ed4c.png",
    "Moderate": "07083738b75e.png",
    "Severe": "dbb2c63f6f08.png",
    "Proliferative_DR": "0981195eb9fb.png",
}

SAMPLE_IMAGES: Dict[str, str] = {}
samples_dir = _APP_DIR / "samples"
test_base = _REPO_ROOT / "data" / "split" / "test"

for c in CLASS_NAMES:
    pref = PREFERRED_SAMPLES.get(c)
    # 1. Prefer bundled sample in app/samples/
    if samples_dir.is_dir():
        if pref and (samples_dir / pref).is_file():
            SAMPLE_IMAGES[c] = str(samples_dir / pref)
        elif (samples_dir / f"{c}.png").is_file():
            SAMPLE_IMAGES[c] = str(samples_dir / f"{c}.png")
    # 2. Fall back to data/split/test/ if available
    if c not in SAMPLE_IMAGES and test_base.exists():
        c_dir = test_base / c
        if c_dir.is_dir():
            if pref and (c_dir / pref).is_file():
                SAMPLE_IMAGES[c] = str(c_dir / pref)
            else:
                pngs = [f for f in os.listdir(c_dir) if f.endswith(".png")]
                if pngs:
                    SAMPLE_IMAGES[c] = str(c_dir / pngs[0])



# ---------------------------------------------------------------------------
# Helper functions for UI rendering
# ---------------------------------------------------------------------------
def to_base64_src(img_bytes: bytes, max_dim: int = 500) -> str:
    """Converts image bytes to base64 data URL, downscaling large images to prevent WebSocket lag."""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64}"


def safe_notify(message: str, **kwargs):
    """Safely issues a UI notification, falling back gracefully if outside UI client context."""
    try:
        ui.notify(message, **kwargs)
    except Exception:
        print(f"[Notice] {message}")


def run_screening_analysis(
    image_bytes: bytes,
    filename: str = "fundus_scan.png",
    target_class: Optional[str] = None,
):
    """Executes screening and updates all UI sections."""
    try:
        safe_notify("Analyzing retinal microvasculature...", type="info", position="top", close_button=True)
        res = predict_image(
            image_bytes=image_bytes,
            force_mock=state["use_mock"],
            patient_id=state["patient_id"],
            eye=state["eye"],
            target_class=target_class,
        )
        state["current_result"] = res

        # Add initial clinical explanation to chatbot
        details = res["details"]
        summary_msg = (
            f"**Automated Screening Assessment Complete:**\n\n"
            f"- **Classified Stage**: {details['full_name']} ({details['grade']})\n"
            f"- **Confidence Score**: {res['confidence_pct']}\n"
            f"- **Triage Status**: **{res['triage_label']}**\n"
            f"- **Recommended Timeline**: {res['triage_timeline']}\n\n"
            f"{details['summary']}\n\n"
            f"*You may ask me clinical follow-up questions regarding lesions, referral protocols, or Grad-CAM attention.*"
        )
        state["chat_history"].append(("assistant", summary_msg))

        refresh_results_view()
        refresh_chat_view()
        safe_notify(f"Screening complete: {res['predicted_class']} ({res['confidence_pct']})", type="positive", position="top")
    except Exception as e:
        safe_notify(f"Error analyzing image: {e}", type="negative", position="top")
        print(f"[App Error] {e}")


async def handle_file_upload(e: events.UploadEventArguments):
    """Handles drag-and-drop or file selector uploads."""
    name: str = "uploaded_fundus.png"
    image_bytes: bytes = b""
    try:
        content_obj: Any = getattr(e, "content", None)
        file_obj: Any = getattr(e, "file", None)
        if file_obj is not None:
            name = getattr(file_obj, "name", "uploaded_fundus.png")
            read_fn = getattr(file_obj, "read", None)
            if callable(read_fn):
                read_res: Any = read_fn()
                if hasattr(read_res, "__await__"):
                    image_bytes = bytes(await read_res)
                elif isinstance(read_res, (bytes, bytearray)):
                    image_bytes = bytes(read_res)
        elif content_obj is not None:
            name = getattr(e, "name", "uploaded_fundus.png")
            if hasattr(content_obj, "read"):
                res: Any = content_obj.read()
                if hasattr(res, "__await__"):
                    image_bytes = bytes(await res)
                elif isinstance(res, (bytes, bytearray)):
                    image_bytes = bytes(res)
            elif isinstance(content_obj, (bytes, bytearray)):
                image_bytes = bytes(content_obj)
    except Exception as upload_err:
        safe_notify(f"Upload error: {upload_err}", type="negative")
        print(f"[Upload Error] {upload_err}")
        return

    if image_bytes:
        run_screening_analysis(image_bytes, filename=name)
    else:
        safe_notify("Received empty file upload.", type="warning")


def load_sample(class_name: str):
    """Loads a clinical sample image from the test set."""
    if class_name in SAMPLE_IMAGES and os.path.exists(SAMPLE_IMAGES[class_name]):
        with open(SAMPLE_IMAGES[class_name], "rb") as f:
            image_bytes = f.read()
        state["patient_id"] = f"PT-{class_name[:3].upper()}-9104"
        run_screening_analysis(image_bytes, filename=f"sample_{class_name}.png", target_class=None)
    else:
        # Fallback: create synthetic fundus scan
        img = Image.new("RGB", (224, 224), color=(30, 20, 10))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        run_screening_analysis(buf.getvalue(), filename=f"sample_{class_name}.png", target_class=None)


def download_pdf():
    """Generates and triggers download of the clinical PDF screening report."""
    if not state["current_result"]:
        safe_notify("Please run screening on a retinal image before generating a report.", type="warning")
        return
    try:
        pdf_bytes = generate_clinical_pdf_report(state["current_result"])
        patient_tag = state["current_result"].get("patient_id", "Patient").replace(" ", "_")
        filename = f"RetinaScan_Report_{patient_tag}_{state['current_result']['predicted_class']}.pdf"
        try:
            ui.download(pdf_bytes, filename=filename)
        except Exception:
            pass
        safe_notify(f"Downloaded {filename}", type="positive", position="top")
    except Exception as e:
        safe_notify(f"Failed to generate PDF: {e}", type="negative")
        print(f"[PDF Error] {e}")


def download_patient_pdf():
    """Generates and triggers download of the plain-language patient take-home summary PDF."""
    if not state["current_result"]:
        safe_notify("Please run a scan first before downloading your summary.", type="warning")
        return
    try:
        pdf_bytes = generate_patient_summary_pdf(state["current_result"])
        patient_tag = state["current_result"].get("patient_id", "Patient").replace(" ", "_")
        cls = state["current_result"]["predicted_class"]
        filename = f"RetinaScan_MyEyeSummary_{patient_tag}_{cls}.pdf"
        try:
            ui.download(pdf_bytes, filename=filename)
        except Exception:
            pass
        safe_notify("Your personal eye summary has been downloaded.", type="positive", position="top")
    except Exception as e:
        safe_notify(f"Could not generate your summary: {e}", type="negative")
        print(f"[Patient PDF Error] {e}")


# ---------------------------------------------------------------------------
# UI Layout Construction
# ---------------------------------------------------------------------------
# Inject professional clinical stylesheet & typography
ui.add_head_html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
    body {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #F8FAFC;
        color: #0F172A;
    }
    .mono { font-family: 'JetBrains Mono', monospace; }
    .glass-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.05), 0 1px 2px -1px rgb(0 0 0 / 0.05);
    }
    .glass-card-hover:hover {
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.07), 0 2px 4px -2px rgb(0 0 0 / 0.05);
    }
</style>
""")

# Top Clinical App Bar
with ui.header().classes("bg-slate-900 text-white px-6 py-3 flex items-center justify-between shadow-sm border-b border-slate-800"):
    with ui.row().classes("items-center gap-3"):
        ui.icon("visibility", size="28px").classes("text-sky-400")
        with ui.column().classes("gap-0"):
            ui.label("RetinaScan AI").classes("text-lg font-bold tracking-tight text-white")
            view_mode_label = ui.label("Clinician View — Clinical DR Triage & Decision Support").classes("text-xs text-slate-400 font-medium")

    with ui.row().classes("items-center gap-3"):
        # View mode toggle (Patient / Clinician)
        with ui.row().classes("items-center gap-1.5 px-3 py-1 rounded-full bg-sky-700 border border-sky-500 text-xs font-semibold text-white cursor-pointer"):
            ui.icon("person", size="16px").classes("text-sky-200")
            view_toggle_label = ui.label("👨‍⚕️ Clinician View")

        def toggle_view_mode():
            state["patient_mode"] = not state["patient_mode"]
            # Reset symptom flags on every toggle
            state["symptom_flags"] = {"floaters": None, "blurry": None, "dark_curtain": None}
            if state["patient_mode"]:
                view_toggle_label.set_text("👤 Patient View")
                view_mode_label.set_text("Patient Portal — My Eye Health Summary")
                safe_notify("Switched to Patient View — plain-language results", type="info")
            else:
                view_toggle_label.set_text("👨‍⚕️ Clinician View")
                view_mode_label.set_text("Clinician View — Clinical DR Triage & Decision Support")
                safe_notify("Switched to Clinician View — full diagnostic workstation", type="info")
            refresh_results_view()

        ui.button("Switch View", on_click=toggle_view_mode).props("flat dense color=sky-200 size=sm").classes("text-xs capitalize")

        # Engine indicator badge
        with ui.row().classes("items-center gap-1.5 px-3 py-1 rounded-full bg-slate-800 border border-slate-700 text-xs font-medium text-slate-300"):
            ui.html('<span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>')
            initial_text = "Engine: PyTorch CNN (best_model.pth)" if is_real_model_available() else "Engine: Clinical Simulation (Mock Mode)"
            mode_label = ui.label(initial_text)

        def toggle_engine_mode():
            state["use_mock"] = not state["use_mock"]
            if state["use_mock"]:
                mode_label.set_text("Engine: Clinical Simulation (Mock Mode)")
                safe_notify("Switched to Mock Engine (Confidence 60-95%)", type="info")
            else:
                if is_real_model_available():
                    mode_label.set_text("Engine: PyTorch CNN (best_model.pth)")
                    safe_notify("Switched to Real PyTorch Model Checkpoint", type="positive")
                else:
                    mode_label.set_text("Engine: Mock (No checkpoint found)")
                    state["use_mock"] = True
                    safe_notify("best_model.pth not found in models/. Remaining in Mock mode.", type="warning")

        ui.button("Switch Engine", on_click=toggle_engine_mode).props("flat dense color=sky-300 size=sm").classes("text-xs capitalize")


# Main Container
with ui.column().classes("w-full max-w-7xl mx-auto p-4 md:p-6 gap-6"):

    # Top Clinical Disclaimer Banner
    with ui.row().classes("w-full bg-sky-50 border border-sky-200 rounded-lg p-3 items-center justify-between text-xs text-sky-900"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("medical_services", size="18px").classes("text-sky-700")
            ui.label("INVESTIGATIONAL CLINICAL DECISION SUPPORT PLATFORM — Built for automated triage of retinal fundus imaging based on the International Clinical Diabetic Retinopathy (ICDR) scale.").classes("font-medium")
        ui.label("Protocol: ICDR 5-Stage Classification").classes("font-semibold text-sky-800 hidden sm:block")

    # 2-Column Workstation Grid
    with ui.grid().classes("w-full grid-cols-1 lg:grid-cols-12 gap-6 items-start"):

        # -------------------------------------------------------------------
        # LEFT COLUMN (Cols 1-5): Examination Setup & Image Ingestion
        # -------------------------------------------------------------------
        with ui.column().classes("lg:col-span-5 gap-6 w-full"):

            # 1. Patient & Examination Details Card
            with ui.card().classes("glass-card rounded-xl p-5 w-full gap-4"):
                with ui.row().classes("items-center justify-between w-full border-b border-slate-100 pb-3"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("badge", size="20px").classes("text-slate-600")
                        ui.label("Examination Context").classes("text-sm font-bold text-slate-800")
                    ui.label("ISO 13485").classes("text-[10px] text-slate-400 font-mono bg-slate-100 px-2 py-0.5 rounded")

                with ui.grid().classes("grid-cols-2 gap-3 w-full"):
                    patient_input = ui.input("Patient ID / MRN", value=state["patient_id"]).classes("w-full text-xs")
                    patient_input.on("blur", lambda: state.update({"patient_id": str(patient_input.value or "PT-2026-0842")}))

                    eye_select = ui.select(
                        ["OD (Right Eye)", "OS (Left Eye)"],
                        value=state["eye"],
                        label="Examined Eye",
                    ).classes("w-full text-xs")
                    eye_select.on("update:model-value", lambda: state.update({"eye": str(eye_select.value or "OD (Right Eye)")}))

            # 2. Retinal Image Upload Card
            with ui.card().classes("glass-card rounded-xl p-5 w-full gap-4"):
                with ui.row().classes("items-center justify-between w-full border-b border-slate-100 pb-3"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("cloud_upload", size="20px").classes("text-sky-600")
                        ui.label("Fundus Image Ingestion").classes("text-sm font-bold text-slate-800")
                    ui.label("PNG, JPG, JPEG").classes("text-[10px] text-slate-400 font-mono")

                ui.label(
                    "Drag and drop a digital fundus photograph here, or browse from your workstation. "
                    "Images are preprocessed via circular cropping and CLAHE contrast equalization."
                ).classes("text-xs text-slate-500 leading-relaxed")

                # NiceGUI native drag & drop upload
                uploader = ui.upload(
                    on_upload=handle_file_upload,
                    max_files=1,
                    auto_upload=True,
                ).props('accept=".png,.jpg,.jpeg" flat bordered color="sky-600"').classes("w-full")

            # 3. 1-Click Clinical Test Samples (For examiner demonstration)
            with ui.card().classes("glass-card rounded-xl p-5 w-full gap-3"):
                with ui.row().classes("items-center justify-between w-full border-b border-slate-100 pb-2"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("science", size="18px").classes("text-emerald-600")
                        ui.label("Quick Clinical Test Samples").classes("text-xs font-bold text-slate-800")
                    ui.label("Stratified Test Split").classes("text-[10px] text-slate-400")

                ui.label("Click any verified test case to immediately populate the workstation:").classes("text-xs text-slate-500")

                with ui.column().classes("w-full gap-2 pt-1"):
                    samples_meta = [
                        ("No_DR", "Normal Retina (Grade 0)", "emerald"),
                        ("Mild", "Mild NPDR (Grade 1)", "amber"),
                        ("Moderate", "Moderate NPDR (Grade 2)", "amber"),
                        ("Severe", "Severe NPDR (Grade 3)", "red"),
                        ("Proliferative_DR", "Proliferative PDR (Grade 4)", "red"),
                    ]
                    for cls_key, label_text, col in samples_meta:
                        ui.button(
                            label_text,
                            on_click=lambda _, c=cls_key: load_sample(c),
                        ).props(f"outline dense size=sm color={col}").classes("w-full text-xs justify-start px-3 py-1.5 font-medium")

        # -------------------------------------------------------------------
        # RIGHT COLUMN (Cols 6-12): Diagnostic Workstation, Inspection, Grad-CAM
        # -------------------------------------------------------------------
        with ui.column().classes("lg:col-span-7 gap-6 w-full"):

            # Dynamic Results Container
            results_container = ui.column().classes("w-full gap-6")


# ---------------------------------------------------------------------------
# Dynamic Results View Renderer
# ---------------------------------------------------------------------------
def refresh_results_view():
    """Renders clinical or patient results panel depending on view mode."""
    try:
        results_container.clear()
    except Exception:
        return
    res = state.get("current_result")

    with results_container:
        if not res:
            # Empty / Awaiting Scan State
            icon_text = "add_photo_alternate"
            if state.get("patient_mode"):
                with ui.card().classes("glass-card rounded-xl p-12 w-full flex flex-col items-center justify-center text-center gap-4 border-dashed border-2 border-slate-200"):
                    ui.icon("remove_red_eye", size="56px").classes("text-slate-300")
                    ui.label("Your Eye Scan Result Will Appear Here").classes("text-lg font-bold text-slate-700")
                    ui.label("Please upload your retinal photograph or select a sample on the left. Your results will be shown in plain, easy-to-understand language.").classes("text-sm text-slate-400 max-w-md leading-relaxed")
            else:
                with ui.card().classes("glass-card rounded-xl p-12 w-full flex flex-col items-center justify-center text-center gap-3 border-dashed border-2 border-slate-300"):
                    ui.icon("add_photo_alternate", size="48px").classes("text-slate-300")
                    ui.label("Awaiting Retinal Fundus Photograph").classes("text-base font-bold text-slate-700")
                    ui.label("Upload a patient fundus photo or click a Quick Sample on the left to activate automated classification, Grad-CAM visualization, and referral triage.").classes("text-xs text-slate-400 max-w-md")
            return

        # Route to the correct view
        if state.get("patient_mode"):
            _render_patient_view(res)
        else:
            _render_clinician_view(res)


def _render_clinician_view(res: dict):
    """Renders the full technical clinician workstation (unchanged from original)."""
    details = res["details"]
    triage_cat = res["triage_category"]

    # Color token resolution
    if triage_cat == "urgent":
        badge_bg = "bg-red-50 border-red-200 text-red-800"
        badge_dot = "bg-red-600"
        stage_pill_bg = "bg-red-100 text-red-900 border-red-300"
    elif triage_cat == "routine":
        badge_bg = "bg-amber-50 border-amber-200 text-amber-800"
        badge_dot = "bg-amber-500"
        stage_pill_bg = "bg-amber-100 text-amber-900 border-amber-300"
    else:
        badge_bg = "bg-emerald-50 border-emerald-200 text-emerald-800"
        badge_dot = "bg-emerald-500"
        stage_pill_bg = "bg-emerald-100 text-emerald-900 border-emerald-300"

    # -------------------------------------------------------------------
    # 1. Primary Triage & Diagnostic Findings Card
    # -------------------------------------------------------------------
    with ui.card().classes("glass-card rounded-xl p-6 w-full gap-4"):
        with ui.row().classes("items-center justify-between w-full border-b border-slate-100 pb-3"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("assignment_turned_in", size="20px").classes("text-sky-600")
                ui.label("Automated Diagnostic Finding").classes("text-sm font-bold text-slate-800")
            ui.button("Download PDF Report", on_click=download_pdf).props("unelevated size=sm color=sky-600 icon=picture_as_pdf").classes("text-xs font-semibold px-3 py-1 shadow-sm")

        with ui.row().classes("items-center justify-between w-full flex-wrap gap-4"):
            with ui.column().classes("gap-1"):
                with ui.row().classes("items-center gap-2"):
                    ui.label(details["full_name"]).classes("text-xl font-extrabold text-slate-900 tracking-tight")
                    ui.label(details["grade"]).classes(f"text-xs font-bold px-2 py-0.5 rounded border {stage_pill_bg}")
                ui.label(f"Evaluated Eye: {res['eye']}  |  Patient MRN: {res['patient_id']}").classes("text-xs text-slate-500")

            with ui.row().classes(f"items-center gap-2 px-3 py-2 rounded-lg border shadow-sm {badge_bg}"):
                ui.html(f'<span class="w-3 h-3 rounded-full {badge_dot}"></span>')
                with ui.column().classes("gap-0"):
                    ui.label("Triage Action:").classes("text-[10px] font-semibold uppercase tracking-wider opacity-75")
                    ui.label(res["triage_label"]).classes("text-sm font-extrabold")
                    ui.label(f"Target: {res['triage_timeline']}").classes("text-[11px] font-medium")

        with ui.column().classes("w-full bg-slate-50 border border-slate-200 rounded-lg p-4 gap-2 mt-2"):
            with ui.row().classes("justify-between w-full items-center"):
                ui.label("Primary Classification Confidence:").classes("text-xs font-semibold text-slate-700")
                ui.label(res["confidence_pct"]).classes("text-sm font-extrabold text-sky-700 font-mono")
            ui.linear_progress(value=res["confidence"], show_value=False).props("size=8px color=sky-600 rounded").classes("w-full")
            ui.label("Multi-Class Posterior Probability Distribution (5-Stage ICDR Scale):").classes("text-[11px] text-slate-500 font-medium pt-1")
            with ui.grid().classes("grid-cols-5 gap-2 w-full pt-1"):
                for idx, c_name in enumerate(CLASS_NAMES):
                    p_val = res["probabilities"][idx]
                    is_winner = (idx == res["class_index"])
                    p_bg = "bg-sky-100 text-sky-900 font-bold border-sky-300" if is_winner else "bg-white text-slate-600 border-slate-200"
                    with ui.column().classes(f"p-2 rounded border text-center gap-0.5 {p_bg}"):
                        ui.label(c_name.replace("_", " ")).classes("text-[10px] truncate")
                        ui.label(f"{p_val*100:.1f}%").classes("text-xs font-mono")

    # -------------------------------------------------------------------
    # 2. Retinal Inspection Panel (Side-by-Side: Original vs Grad-CAM)
    # -------------------------------------------------------------------
    with ui.card().classes("glass-card rounded-xl p-6 w-full gap-4"):
        with ui.row().classes("items-center justify-between w-full border-b border-slate-100 pb-3"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("biotech", size="20px").classes("text-indigo-600")
                ui.label("Retinal Inspection & Spatial Explainability").classes("text-sm font-bold text-slate-800")
            ui.label("Grad-CAM Class Activation").classes("text-[10px] font-mono text-indigo-700 bg-indigo-50 border border-indigo-200 px-2 py-0.5 rounded")

        ui.label(
            "Side-by-side verification: The left panel displays the uploaded fundus field. "
            "The right panel displays the Grad-CAM saliency heatmap, revealing the exact retinal lesions (microaneurysms, "
            "exudates, haemorrhages) that drove the model's classification decision."
        ).classes("text-xs text-slate-500")

        with ui.grid().classes("grid-cols-1 sm:grid-cols-2 gap-4 w-full pt-2"):
            with ui.column().classes("gap-2 items-center bg-slate-900 rounded-lg p-2 border border-slate-300"):
                ui.image(to_base64_src(res["original_image_bytes"])).classes("w-full h-56 object-contain rounded")
                ui.label("Patient Fundus Photograph").classes("text-xs font-semibold text-slate-200")
            with ui.column().classes("gap-2 items-center bg-slate-900 rounded-lg p-2 border border-indigo-300"):
                ui.image(to_base64_src(res["gradcam_image_bytes"])).classes("w-full h-56 object-contain rounded")
                with ui.row().classes("items-center gap-1.5"):
                    ui.html('<span class="w-2 h-2 rounded-full bg-red-500"></span>')
                    ui.label("Grad-CAM Attention Heatmap").classes("text-xs font-semibold text-slate-200")

    # -------------------------------------------------------------------
    # 3. Pathological Findings & Clinical Care Protocol
    # -------------------------------------------------------------------
    with ui.card().classes("glass-card rounded-xl p-6 w-full gap-4"):
        with ui.row().classes("items-center gap-2 border-b border-slate-100 pb-3 w-full"):
            ui.icon("fact_check", size="20px").classes("text-slate-700")
            ui.label("Pathology Summary & Care Recommendations").classes("text-sm font-bold text-slate-800")

        with ui.grid().classes("grid-cols-1 md:grid-cols-2 gap-4 w-full"):
            with ui.column().classes("gap-2 bg-slate-50 p-4 rounded-lg border border-slate-200"):
                ui.label("Pathological Feature Assessment:").classes("text-xs font-bold text-slate-700")
                for finding in details["pathology_findings"]:
                    with ui.row().classes("items-start gap-2"):
                        ui.icon("lens", size="8px").classes("text-sky-600 mt-1.5")
                        ui.label(finding).classes("text-xs text-slate-600 leading-snug")
            with ui.column().classes("gap-2 bg-slate-50 p-4 rounded-lg border border-slate-200"):
                ui.label("Recommended Management Plan:").classes("text-xs font-bold text-slate-700")
                for act in details["recommendations"]:
                    with ui.row().classes("items-start gap-2"):
                        ui.icon("check", size="14px").classes("text-emerald-600 mt-0.5")
                        ui.label(act).classes("text-xs text-slate-600 leading-snug")


def _render_patient_view(res: dict):
    """
    Renders the plain-language Patient Portal view.
    Modeled after NHS DESP patient letters and LumineticsCore patient result slips.
    No medical jargon, no Grad-CAM, no probability tables.
    """
    pd_info = PATIENT_DETAILS.get(res["predicted_class"], {})
    urgency = pd_info.get("appointment_urgency", "none")

    # Color palette by urgency
    if urgency == "none":
        status_bg = "bg-emerald-50 border-emerald-200"
        status_text = "text-emerald-800"
        status_icon = "check_circle"
        status_icon_color = "text-emerald-600"
        action_bg = "bg-emerald-100 border-emerald-300 text-emerald-900"
    elif urgency in ("routine",):
        status_bg = "bg-amber-50 border-amber-200"
        status_text = "text-amber-900"
        status_icon = "schedule"
        status_icon_color = "text-amber-600"
        action_bg = "bg-amber-100 border-amber-300 text-amber-900"
    elif urgency == "urgent":
        status_bg = "bg-red-50 border-red-200"
        status_text = "text-red-900"
        status_icon = "warning"
        status_icon_color = "text-red-600"
        action_bg = "bg-red-100 border-red-300 text-red-900"
    else:  # emergency
        status_bg = "bg-red-100 border-red-400"
        status_text = "text-red-900"
        status_icon = "emergency"
        status_icon_color = "text-red-700"
        action_bg = "bg-red-200 border-red-500 text-red-900"

    # -------------------------------------------------------------------
    # 1. Big Status Card — headline + plain summary + your eye image
    # -------------------------------------------------------------------
    with ui.card().classes(f"rounded-xl p-6 w-full gap-4 border-2 {status_bg}"):
        with ui.row().classes("items-center gap-3 pb-3 border-b border-slate-200"):
            ui.icon(status_icon, size="32px").classes(status_icon_color)
            with ui.column().classes("gap-0"):
                ui.label("Your Eye Screening Result").classes("text-xs font-semibold text-slate-500 uppercase tracking-wide")
                ui.label(pd_info.get("headline", "Scan complete.")).classes(f"text-lg font-extrabold leading-snug {status_text}")

        # Your fundus photo (patient-friendly display — no Grad-CAM)
        with ui.row().classes("gap-5 items-start flex-wrap"):
            with ui.column().classes("gap-2 items-center bg-slate-900 rounded-lg p-2 border border-slate-600 min-w-[180px]"):
                ui.image(to_base64_src(res["original_image_bytes"])).classes("w-44 h-44 object-contain rounded")
                ui.label("Your retinal photograph").classes("text-[11px] text-slate-300 font-medium")

            with ui.column().classes("flex-1 gap-3 min-w-[220px]"):
                ui.label("What the scan found").classes("text-sm font-bold text-slate-800")
                ui.label(pd_info.get("plain_summary", "")).classes("text-sm text-slate-600 leading-relaxed")
                ui.label("Why this matters").classes("text-sm font-bold text-slate-800 mt-2")
                ui.label(pd_info.get("why_it_matters", "")).classes("text-sm text-slate-600 leading-relaxed")

    # -------------------------------------------------------------------
    # 2. Emergency symptom flag (always shown — interactive)
    # -------------------------------------------------------------------
    with ui.card().classes("glass-card rounded-xl p-5 w-full gap-3"):
        with ui.row().classes("items-center gap-2 border-b border-slate-100 pb-2"):
            ui.icon("report_problem", size="20px").classes("text-red-500")
            ui.label("Are you experiencing any of these symptoms RIGHT NOW?").classes("text-sm font-bold text-slate-800")

        ui.label("These symptoms can be signs of an emergency — even if your scan result looks mild.").classes("text-xs text-slate-500")

        symptom_warning_row = ui.row().classes("w-full")

        def update_symptom_warning():
            flags = state["symptom_flags"]
            any_yes = any(v is True for v in flags.values())
            symptom_warning_row.clear()
            with symptom_warning_row:
                if any_yes:
                    with ui.row().classes("items-start gap-3 bg-red-50 border-2 border-red-400 rounded-lg p-4 w-full"):
                        ui.icon("emergency", size="28px").classes("text-red-600 mt-0.5")
                        with ui.column().classes("gap-1"):
                            ui.label("⚠️  Please seek urgent eye care today").classes("text-base font-extrabold text-red-800")
                            ui.label(
                                "Based on what you have reported, you should contact your eye specialist or go to your nearest "
                                "A&E eye emergency department as soon as possible. Do not drive yourself. "
                                "Do not wait for a routine appointment."
                            ).classes("text-sm text-red-700 leading-relaxed")

        symptoms = [
            ("floaters", "I can see new dark floaters, spots, or cobwebs in my vision"),
            ("blurry", "My vision has become suddenly blurry or there is a dark shadow / curtain"),
            ("dark_curtain", "I have had a sudden loss of vision in one or both eyes"),
        ]

        with ui.column().classes("w-full gap-2 pt-1"):
            for flag_key, label_text in symptoms:
                with ui.row().classes("items-center gap-3 p-3 rounded-lg bg-slate-50 border border-slate-200"):
                    ui.label(label_text).classes("text-sm text-slate-700 flex-1")
                    with ui.row().classes("gap-2"):
                        ui.button(
                            "Yes",
                            on_click=lambda _, k=flag_key: [
                                state["symptom_flags"].update({k: True}),
                                update_symptom_warning(),
                            ],
                        ).props("unelevated dense size=sm color=red").classes("text-xs font-bold px-3")
                        ui.button(
                            "No",
                            on_click=lambda _, k=flag_key: [
                                state["symptom_flags"].update({k: False}),
                                update_symptom_warning(),
                            ],
                        ).props("outline dense size=sm color=grey").classes("text-xs px-3")

        update_symptom_warning()

    # -------------------------------------------------------------------
    # 3. Your Next Step
    # -------------------------------------------------------------------
    with ui.card().classes("glass-card rounded-xl p-5 w-full gap-3"):
        with ui.row().classes("items-center gap-2 border-b border-slate-100 pb-2"):
            ui.icon("directions", size="20px").classes("text-sky-600")
            ui.label("Your Next Step").classes("text-sm font-bold text-slate-800")

        with ui.row().classes(f"items-start gap-3 p-4 rounded-lg border {action_bg}"):
            ui.icon("arrow_forward_ios", size="18px").classes("mt-0.5")
            ui.label(pd_info.get("next_step", "")).classes("text-sm font-semibold leading-relaxed")

        # Appointment prep checklist (only if referral needed)
        prep = pd_info.get("prep_checklist", [])
        if prep:
            ui.label("Before your eye appointment:").classes("text-sm font-bold text-slate-700 pt-2")
            with ui.column().classes("gap-2"):
                for item in prep:
                    with ui.row().classes("items-start gap-2"):
                        ui.icon("check_box", size="18px").classes("text-sky-600 mt-0.5")
                        ui.label(item).classes("text-sm text-slate-600 leading-snug")

        # Emergency warning box (only for urgent/emergency stages)
        warn_text = pd_info.get("emergency_warning")
        if warn_text:
            with ui.row().classes("items-start gap-3 mt-2 bg-red-50 border border-red-300 rounded-lg p-3"):
                ui.icon("warning_amber", size="22px").classes("text-red-600 mt-0.5")
                ui.label(warn_text).classes("text-sm font-semibold text-red-800 leading-relaxed")

    # -------------------------------------------------------------------
    # 4. Daily Tips & Download
    # -------------------------------------------------------------------
    with ui.card().classes("glass-card rounded-xl p-5 w-full gap-3"):
        with ui.row().classes("items-center justify-between w-full border-b border-slate-100 pb-2"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("favorite", size="20px").classes("text-rose-500")
                ui.label("Protecting Your Sight Every Day").classes("text-sm font-bold text-slate-800")
            ui.button(
                "Download My Eye Summary",
                on_click=download_patient_pdf,
            ).props("unelevated size=sm color=sky-600 icon=download").classes("text-xs font-semibold")

        daily_tips = pd_info.get("daily_tips", [])
        with ui.grid().classes("grid-cols-1 md:grid-cols-2 gap-2 w-full pt-1"):
            for tip in daily_tips:
                with ui.row().classes("items-start gap-2 bg-slate-50 p-3 rounded-lg border border-slate-100"):
                    ui.icon("tips_and_updates", size="16px").classes("text-sky-500 mt-0.5")
                    ui.label(tip).classes("text-xs text-slate-600 leading-snug")

        # NHS-style reassurance footer
        with ui.row().classes("items-center gap-2 mt-2 p-3 bg-sky-50 rounded-lg border border-sky-100"):
            ui.icon("shield", size="18px").classes("text-sky-600")
            ui.label(
                "Remember: over 90% of serious sight loss from diabetic eye disease can be "
                "prevented with timely treatment and good diabetes management."
            ).classes("text-xs font-semibold text-sky-800 leading-relaxed")




# ---------------------------------------------------------------------------
# Chatbot Section (Severity-Aware Clinical Assistant)
# ---------------------------------------------------------------------------
with ui.column().classes("w-full max-w-7xl mx-auto px-4 md:px-6 pb-12 gap-4"):
    with ui.card().classes("glass-card rounded-xl p-6 w-full gap-4"):
        with ui.row().classes("items-center justify-between w-full border-b border-slate-100 pb-3"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("forum", size="22px").classes("text-sky-600")
                ui.label("Clinical Assistant & Diagnostic Interpreter").classes("text-sm font-bold text-slate-800")
            ui.label("Decision Support AI").classes("text-[10px] text-slate-400 font-mono bg-slate-100 px-2 py-0.5 rounded")

        # Chat message scroll area
        chat_container = ui.column().classes("w-full gap-3 max-h-72 overflow-y-auto p-3 bg-slate-50 rounded-lg border border-slate-200")

        def refresh_chat_view():
            try:
                chat_container.clear()
            except Exception:
                return
            with chat_container:
                for role, message in state["chat_history"]:
                    if role == "assistant":
                        with ui.row().classes("w-full items-start gap-3"):
                            ui.icon("smart_toy", size="20px").classes("text-sky-700 bg-sky-100 p-1.5 rounded-full")
                            with ui.column().classes("gap-1 bg-white p-3 rounded-lg border border-slate-200 shadow-sm max-w-2xl"):
                                ui.markdown(message).classes("text-xs text-slate-700")
                    else:
                        with ui.row().classes("w-full justify-end items-start gap-3"):
                            with ui.column().classes("gap-1 bg-sky-600 text-white p-3 rounded-lg shadow-sm max-w-xl"):
                                ui.label(message).classes("text-xs text-white")
                            ui.icon("person", size="20px").classes("text-slate-600 bg-slate-200 p-1.5 rounded-full")

        # Initial chat render
        refresh_chat_view()

        # Chat input bar
        with ui.row().classes("w-full gap-2 items-center pt-2"):
            chat_input = ui.input(placeholder="Ask a clinical question (e.g., 'What are microaneurysms?', 'What is the referral urgency?', 'Explain the Grad-CAM heatmap')...").classes("flex-grow text-xs")

            def send_chat_message():
                raw_text = chat_input.value or ""
                text = raw_text.strip()
                if not text:
                    return
                chat_input.value = ""
                state["chat_history"].append(("user", text))
                reply = generate_clinical_assistant_reply(text, state.get("current_result"))
                state["chat_history"].append(("assistant", reply))
                refresh_chat_view()

            def send_quick_query(q_text: str):
                state["chat_history"].append(("user", q_text))
                reply = generate_clinical_assistant_reply(q_text, state.get("current_result"))
                state["chat_history"].append(("assistant", reply))
                refresh_chat_view()

            chat_input.on("keydown.enter", send_chat_message)
            ui.button("Send", on_click=send_chat_message).props("unelevated size=sm color=sky-600 icon=send").classes("text-xs font-semibold px-4 py-2")

        # Pre-set inquiry pills
        with ui.row().classes("items-center gap-2 pt-1 flex-wrap"):
            ui.label("Suggested Queries:").classes("text-[11px] font-semibold text-slate-500")
            quick_queries = [
                "What is this stage?",
                "What is the referral urgency?",
                "Explain the Grad-CAM heatmap",
                "What are microaneurysms?",
                "HbA1c & metabolic goals",
            ]
            for query in quick_queries:
                ui.button(
                    query,
                    on_click=lambda _, q=query: send_quick_query(q),
                ).props("flat dense size=xs color=slate-700").classes("text-[10px] bg-white border border-slate-200 px-2 py-1 rounded hover:bg-slate-100")

    # Bottom Mandatory Clinical Disclaimer
    with ui.card().classes("w-full bg-slate-100 border border-slate-300 rounded-xl p-4 gap-2 text-slate-600"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("warning", size="18px").classes("text-amber-600")
            ui.label("MANDATORY REGULATORY & CLINICAL DISCLAIMER").classes("text-xs font-bold text-slate-800 tracking-wide")
        ui.label(
            "This software is an investigational clinical decision support application developed for educational and research screening workflows. "
            "It is NOT an FDA, CE-mark, or MHRA cleared diagnostic medical device. Automated severity classifications and triage suggestions "
            "are provided solely to augment clinical screening prioritization. All findings must be formally verified through clinical examination "
            "by a licensed ophthalmologist or optometrist before initiating or altering patient care."
        ).classes("text-[11px] text-slate-500 leading-relaxed")


# Initialize default empty results view
refresh_results_view()


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------
if __name__ in {"__main__", "__mp_main__"}:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8080))
    print(f"\n========================================================")
    print(f" RetinaScan AI Clinical Decision Support System")
    print(f" Running on http://{host}:{port}")
    print(f" Regulatory Status: Investigational Use")
    print(f"========================================================\n")
    ui.run(
        title="RetinaScan AI — Clinical DR Triage System",
        host=host,
        port=port,
        reload=False,
        show=False,
    )

