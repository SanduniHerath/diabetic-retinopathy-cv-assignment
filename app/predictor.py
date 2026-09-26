"""
app/predictor.py
================
Inference and Clinical Triage Engine for Diabetic Retinopathy Screening.

This module provides a modular prediction pipeline:
  1. Default Mock Mode: Returns clinically-sound predictions with random confidence (60-95%),
     multinomial probability distribution, and synthetic Grad-CAM heatmap overlays.
  2. Real Model Mode: Easily enabled by pointing to a saved PyTorch checkpoint (best_model.pth),
     seamlessly executing the 5-step Phase 2 preprocessing pipeline, model inference,
     and Phase 4 Grad-CAM explainability.

Author: Sanduni Herath
Repository: https://github.com/SanduniHerath/diabetic-retinopathy-cv-assignment
"""

import io
import os
import sys
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Portable import setup for repo root and src/
_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
for _p in (_APP_DIR, _REPO_ROOT, _REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Clinical Class Definitions & Severity Standards (ICDR Scale)
# ---------------------------------------------------------------------------
CLASS_NAMES: List[str] = [
    "No_DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative_DR",
]

CLINICAL_DETAILS: Dict[str, Dict[str, Any]] = {
    "No_DR": {
        "grade": "Grade 0",
        "full_name": "No Apparent Diabetic Retinopathy",
        "triage_label": "No Referral Needed",
        "triage_timeline": "Annual Routine Rescreening",
        "triage_category": "none",  # Green
        "color": "#059669",         # Emerald-600
        "bg_color": "#ECFDF5",      # Emerald-50
        "border_color": "#A7F3D0",  # Emerald-200
        "text_color": "#065F46",    # Emerald-800
        "badge_icon": "check_circle",
        "summary": (
            "No diabetic retinopathy lesions detected. Retinal vasculature, optic disc margins, "
            "and foveal avascular zone appear physiologically normal with zero visible microaneurysms or haemorrhages."
        ),
        "pathology_findings": [
            "Normal optic disc contour and cup-to-disc ratio",
            "Clear background without microaneurysms or hard exudates",
            "Regular arteriolar and venular calibres without beading",
            "Sharp macula and intact foveal reflex",
        ],
        "recommendations": [
            "Maintain annual tele-retinal or dilated fundus examination schedule.",
            "Continue optimal glycaemic regulation (HbA1c target typically < 7.0%).",
            "Standard blood pressure and serum lipid monitoring with primary physician.",
            "Advise patient to report any acute visual floaters, flashes, or central blurring immediately.",
        ],
    },
    "Mild": {
        "grade": "Grade 1",
        "full_name": "Mild Non-Proliferative Diabetic Retinopathy (NPDR)",
        "triage_label": "Routine Follow-up",
        "triage_timeline": "Ophthalmology Review in 6–12 Months",
        "triage_category": "routine",  # Amber/Yellow
        "color": "#D97706",            # Amber-600
        "bg_color": "#FFFBEB",         # Amber-50
        "border_color": "#FDE68A",     # Amber-200
        "text_color": "#92400E",       # Amber-800
        "badge_icon": "info",
        "summary": (
            "Isolated microaneurysms detected in the retinal capillary beds. Early microvascular compromise "
            "consistent with early-stage non-proliferative diabetic eye disease."
        ),
        "pathology_findings": [
            "Presence of isolated microaneurysms (focal capillary outpouchings)",
            "Absence of hard exudates, cotton wool spots, or venous abnormalities",
            "Macular area remains clear of clinically significant macular edema",
            "No neovascular proliferation detected",
        ],
        "recommendations": [
            "Schedule non-urgent ophthalmology / optometry dilated review within 6 to 12 months.",
            "Intensify diabetic metabolic management in coordination with endocrinologist.",
            "Optimize systemic blood pressure (< 130/80 mmHg) to retard microvascular progression.",
            "Educate patient regarding the asymptomatic nature of early diabetic eye disease.",
        ],
    },
    "Moderate": {
        "grade": "Grade 2",
        "full_name": "Moderate Non-Proliferative Diabetic Retinopathy (NPDR)",
        "triage_label": "Routine Follow-up",
        "triage_timeline": "Ophthalmologist Consultation within 3–6 Months",
        "triage_category": "routine",  # Amber/Yellow
        "color": "#D97706",            # Amber-600
        "bg_color": "#FFFBEB",         # Amber-50
        "border_color": "#FDE68A",     # Amber-200
        "text_color": "#92400E",       # Amber-800
        "badge_icon": "schedule",
        "summary": (
            "Multiple microaneurysms accompanied by intraretinal blot haemorrhages and/or hard lipid exudates. "
            "Microvascular abnormalities exceed mild criteria but do not yet meet the 4:2:1 severe threshold."
        ),
        "pathology_findings": [
            "Widespread microaneurysms and dot-and-blot intraretinal haemorrhages",
            "Hard exudates (lipid residues) indicating localized vascular hyperpermeability",
            "Possible minor cotton wool spots (nerve fibre layer infarcts)",
            "Risk of concurrent diabetic macular edema (DME) requires clinical biomicroscopy",
        ],
        "recommendations": [
            "Refer for comprehensive slit-lamp biomicroscopy and Optical Coherence Tomography (OCT) within 3 to 6 months.",
            "Assess for subclinical macular thickening or central involvement.",
            "Review renal panel (microalbuminuria) and lipid profiles as systemic vascular markers.",
            "Structured patient counselling regarding lifestyle and strict glycaemic targets.",
        ],
    },
    "Severe": {
        "grade": "Grade 3",
        "full_name": "Severe Non-Proliferative Diabetic Retinopathy (NPDR)",
        "triage_label": "Refer Urgently",
        "triage_timeline": "Urgent Ophthalmology Referral within 2–4 Weeks",
        "triage_category": "urgent",  # Red
        "color": "#DC2626",           # Red-600
        "bg_color": "#FEF2F2",        # Red-50
        "border_color": "#FECACA",    # Red-200
        "text_color": "#991B1B",      # Red-800
        "badge_icon": "warning",
        "summary": (
            "Extensive retinal ischaemia indicated by the International Clinical '4:2:1' rule criteria. "
            "High near-term risk of progression to proliferative disease and irreversible sight loss without intervention."
        ),
        "pathology_findings": [
            "Severe intraretinal haemorrhages in all 4 retinal quadrants",
            "Definite venous beading in 2 or more quadrants",
            "Prominent Intraretinal Microvascular Abnormalities (IRMA) in >= 1 quadrant",
            "High risk of rapid progression to proliferative neovascularisation within 12 months (~50%)",
        ],
        "recommendations": [
            "Urgent retina specialist consultation required within 2 to 4 weeks.",
            "Perform macular OCT and consider wide-field fundus fluorescein angiography (FFA).",
            "Evaluate candidacy for prophylactic panretinal photocoagulation (PRP) or anti-VEGF therapy.",
            "Instruct patient on immediate emergency contact if sudden dark floaters or visual drop occur.",
        ],
    },
    "Proliferative_DR": {
        "grade": "Grade 4",
        "full_name": "Proliferative Diabetic Retinopathy (PDR)",
        "triage_label": "Refer Urgently",
        "triage_timeline": "Immediate Referral (Within 24–48 Hours / Emergency)",
        "triage_category": "urgent",  # Red
        "color": "#B91C1C",           # Dark Red
        "bg_color": "#FEF2F2",        # Red-50
        "border_color": "#FCA5A5",    # Red-300
        "text_color": "#7F1D1D",      # Dark Red-900
        "badge_icon": "error",
        "summary": (
            "Advanced, sight-threatening diabetic retinopathy characterized by pathological neovascularisation "
            "(new fragile vessels). Extreme risk of vitreous haemorrhage, tractional retinal detachment, and permanent blindness."
        ),
        "pathology_findings": [
            "Neovascularisation of the disc (NVD) or elsewhere on the retina (NVE)",
            "Preretinal / subhyaloid or vitreous haemorrhage risk",
            "Fibrovascular proliferation with potential traction on the retinal surface",
            "Imminent threat to central vision demanding urgent secondary eye care",
        ],
        "recommendations": [
            "Immediate referral to vitreoretinal surgeon or ophthalmology emergency clinic (within 24–48 hours).",
            "Immediate Panretinal Photocoagulation (PRP) laser or intravitreal anti-VEGF injection evaluation.",
            "Advise patient against heavy lifting, strenuous exertion, or head-down postures to avoid vitreous bleed.",
            "Comprehensive systemic assessment with diabetes specialist team.",
        ],
    },
}

# ---------------------------------------------------------------------------
# Configuration: Toggle between Mock Prediction and Real Model Checkpoint
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = _REPO_ROOT / "reports" / "training" / "best_model.pth"
_LOADED_MODEL = None
_LOADED_DEVICE = None


def is_real_model_available() -> bool:
    """Checks whether a valid trained PyTorch checkpoint is present on disk."""
    return DEFAULT_CHECKPOINT.is_file() and DEFAULT_CHECKPOINT.stat().st_size > 1_000_000


def get_real_model():
    """Loads and caches the real PyTorch model checkpoint when available."""
    global _LOADED_MODEL, _LOADED_DEVICE
    if _LOADED_MODEL is not None:
        return _LOADED_MODEL, _LOADED_DEVICE

    if not is_real_model_available():
        return None, None

    try:
        import torch
        from model import build_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(backbone_name="efficientnet_b0")
        state = torch.load(str(DEFAULT_CHECKPOINT), map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        _LOADED_MODEL = model
        _LOADED_DEVICE = device
        return _LOADED_MODEL, _LOADED_DEVICE
    except Exception as e:
        print(f"[Predictor] Note: Could not load real model checkpoint: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Heatmap Synthesis (Realistic Clinical Attention for Mock & Explainability)
# ---------------------------------------------------------------------------
def generate_synthetic_gradcam(image_rgb: np.ndarray, class_name: str) -> np.ndarray:
    """
    Synthesizes a realistic Grad-CAM visual attention heatmap over the fundus photo
    based on the predicted clinical stage (e.g. focusing on vessels, macula, or lesions).
    """
    h, w = image_rgb.shape[:2]
    # Work on a standard 300x300 working canvas
    canvas_size = 300
    resized_rgb = cv2.resize(image_rgb, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)

    # Base attention grid
    y, x = np.ogrid[:canvas_size, :canvas_size]
    cx, cy = canvas_size // 2, canvas_size // 2

    # Anatomical lesion center points
    heat_accum = np.zeros((canvas_size, canvas_size), dtype=np.float32)

    if class_name == "No_DR":
        # Broad, low-intensity diffuse attention over the general retinal vascular arcade
        mask = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * 80**2))
        heat_accum += mask * 0.35
    elif class_name == "Mild":
        # Small focal attention spots representing isolated microaneurysms
        spots = [(cx - 45, cy + 25), (cx + 50, cy - 30)]
        for sx, sy in spots:
            mask = np.exp(-((x - sx)**2 + (y - sy)**2) / (2 * 18**2))
            heat_accum += mask * 0.85
    elif class_name == "Moderate":
        # Multiple moderate foci in temporal arcade and macular margin
        spots = [(cx - 55, cy + 30), (cx + 35, cy + 45), (cx - 20, cy - 50)]
        for sx, sy in spots:
            mask = np.exp(-((x - sx)**2 + (y - sy)**2) / (2 * 28**2))
            heat_accum += mask * 0.80
    elif class_name == "Severe":
        # Multi-quadrant blot haemorrhage hotspots
        spots = [(cx - 60, cy - 40), (cx + 60, cy - 40), (cx - 50, cy + 50), (cx + 55, cy + 45)]
        for sx, sy in spots:
            mask = np.exp(-((x - sx)**2 + (y - sy)**2) / (2 * 32**2))
            heat_accum += mask * 0.85
    else:  # Proliferative_DR
        # Intense focal hotspot at optic disc (NVD) and peripheral fronds
        spots = [(cx - 45, cy), (cx + 20, cy - 35), (cx + 70, cy + 20)]
        for sx, sy in spots:
            mask = np.exp(-((x - sx)**2 + (y - sy)**2) / (2 * 25**2))
            heat_accum += mask * 0.95

    # Normalize to 0-255 uint8
    heat_norm = np.clip(heat_accum / (heat_accum.max() + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    color_map = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    color_map_rgb = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

    # Alpha overlay
    overlay = cv2.addWeighted(resized_rgb, 0.62, color_map_rgb, 0.38, 0)
    # Resize back to original dimensions
    final_overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_LINEAR)
    return final_overlay


# ---------------------------------------------------------------------------
# Prediction Engine
# ---------------------------------------------------------------------------
def predict_image(
    image_bytes: bytes,
    force_mock: bool = True,
    patient_id: str = "PT-8291",
    eye: str = "OD (Right Eye)",
    target_class: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Core prediction dispatcher.

    Args:
        image_bytes: Raw binary bytes of uploaded image.
        force_mock: If True, uses the realistic mock engine (default during development).
        patient_id: Clinical patient identifier.
        eye: Examined eye notation (OD/OS).
        target_class: Optional pre-set class to simulate (e.g. for clinical sample buttons).

    Returns:
        Structured clinical screening results dictionary.
    """
    # 1. Decode image bytes to PIL and NumPy RGB
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_image_rgb = np.array(pil_image)

    # 2. Check if real model should be used
    real_model, device = (None, None) if force_mock else get_real_model()

    if real_model is not None and not force_mock:
        # Real model inference path
        try:
            import torch
            import torch.nn.functional as F
            from train import get_val_transforms
            from gradcam import generate_gradcam_heatmap

            transform = get_val_transforms()
            tensor = transform(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = real_model(tensor)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy().tolist()

            pred_idx = int(np.argmax(probs))
            predicted_class = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx])

            # Generate real Grad-CAM heatmap
            heatmap_raw = generate_gradcam_heatmap(real_model, tensor, target_class=pred_idx)
            h, w = np_image_rgb.shape[:2]
            heatmap_res = cv2.resize(heatmap_raw, (w, h))
            heatmap_col = cv2.applyColorMap((heatmap_res * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_col, cv2.COLOR_BGR2RGB)
            gradcam_rgb = cv2.addWeighted(np_image_rgb, 0.60, heatmap_rgb, 0.40, 0)
            engine_mode = "PyTorch Deep CNN (EfficientNet-B0 Staged Fine-Tuned)"
        except Exception as e:
            print(f"[Predictor] Fallback to mock inference due to: {e}")
            real_model = None

    if real_model is None:
        # Mock prediction path (as requested by user requirements)
        # Select specified target_class or random class from CLASS_NAMES
        if target_class and target_class in CLASS_NAMES:
            pred_idx = CLASS_NAMES.index(target_class)
        else:
            pred_idx = random.randint(0, len(CLASS_NAMES) - 1)
        predicted_class = CLASS_NAMES[pred_idx]

        # Generate confidence between 60.0% and 95.0%
        primary_conf = round(random.uniform(0.68, 0.94), 4)

        # Distribute remaining probability across other classes
        remaining = 1.0 - primary_conf
        noise = [random.uniform(0.01, 1.0) for _ in range(len(CLASS_NAMES) - 1)]
        noise_sum = sum(noise)
        norm_noise = [round(n / noise_sum * remaining, 4) for n in noise]

        probs = []
        noise_idx = 0
        for i in range(len(CLASS_NAMES)):
            if i == pred_idx:
                probs.append(primary_conf)
            else:
                probs.append(norm_noise[noise_idx])
                noise_idx += 1

        confidence = primary_conf
        gradcam_rgb = generate_synthetic_gradcam(np_image_rgb, predicted_class)
        engine_mode = "Clinical Diagnostic Simulation Engine (Pre-deployment Validation Mode)"

    # Format Grad-CAM image as PNG bytes
    gradcam_buf = io.BytesIO()
    Image.fromarray(gradcam_rgb).save(gradcam_buf, format="PNG")
    gradcam_bytes = gradcam_buf.getvalue()

    # Retrieve clinical guidelines
    details = CLINICAL_DETAILS[predicted_class]

    return {
        "patient_id": patient_id,
        "eye": eye,
        "predicted_class": predicted_class,
        "class_index": pred_idx,
        "confidence": confidence,
        "confidence_pct": f"{confidence * 100:.1f}%",
        "probabilities": [round(float(p), 4) for p in probs],
        "class_names": CLASS_NAMES,
        "details": details,
        "triage_label": details["triage_label"],
        "triage_timeline": details["triage_timeline"],
        "triage_category": details["triage_category"],
        "triage_color": details["color"],
        "triage_bg_color": details["bg_color"],
        "triage_text_color": details["text_color"],
        "summary": details["summary"],
        "pathology_findings": details["pathology_findings"],
        "recommendations": details["recommendations"],
        "original_image_bytes": image_bytes,
        "gradcam_image_bytes": gradcam_bytes,
        "engine_mode": engine_mode,
    }


# ---------------------------------------------------------------------------
# Clinical Assistant Chatbot Response Engine
# ---------------------------------------------------------------------------
def generate_clinical_assistant_reply(user_query: str, current_result: Optional[Dict[str, Any]]) -> str:
    """
    Severity-aware clinical assistant logic that answers medical and screening questions
    based on the currently evaluated patient and retinal findings.
    """
    q = (user_query or "").strip().lower()

    if not current_result:
        return (
            "Please upload a retinal fundus photograph first. Once evaluated, I will provide "
            "clinical context, feature interpretations, and referral guidance tailored to the patient's examination."
        )

    pred_class = current_result["predicted_class"]
    details = current_result["details"]
    conf = current_result["confidence_pct"]

    if any(k in q for k in ["stage", "grade", "diagnosis", "result", "what is"]):
        return (
            f"**Automated Finding: {details['full_name']} ({details['grade']})**\n\n"
            f"- **Detection Confidence**: {conf}\n"
            f"- **Clinical Summary**: {details['summary']}\n"
            f"- **Triage Priority**: **{details['triage_label']}** ({details['triage_timeline']})"
        )

    elif any(k in q for k in ["refer", "triage", "next step", "timeline", "when", "action"]):
        actions = "\n".join([f"  {idx+1}. {act}" for idx, act in enumerate(details["recommendations"])])
        return (
            f"**Recommended Referral & Management Protocol for {details['grade']}:**\n\n"
            f"**Triage Action**: {details['triage_label']} ({details['triage_timeline']})\n\n"
            f"**Action Plan**:\n{actions}"
        )

    elif any(k in q for k in ["gradcam", "heatmap", "attention", "yellow", "red area"]):
        return (
            "**Grad-CAM Explainability Interpretation:**\n\n"
            "The Grad-CAM (Gradient-Weighted Class Activation Mapping) overlay highlights the spatial "
            "regions that most strongly influenced the neural network's classification. "
            "Warmer colors (red, orange, yellow) indicate high diagnostic saliency—typically aligning with "
            "microaneurysms, vascular tortuosity, exudate clusters, or retinal neovascularisation. "
            "Cooler regions (blue/cyan) represent background anatomical structures that exerted minimal influence."
        )

    elif any(k in q for k in ["microaneurysm", "exudate", "haemorrhage", "cotton wool", "symptom"]):
        return (
            "**Diabetic Retinopathy Biomarkers:**\n\n"
            "- **Microaneurysms**: Focal outpouchings of capillary walls; the earliest visible sign of diabetic retinal microangiopathy.\n"
            "- **Hard Exudates**: Lipid and lipoprotein precipitates leaking from abnormally permeable capillaries.\n"
            "- **Hemorrhages (Dot/Blot)**: Bleeding located deep in the inner nuclear retinal layer.\n"
            "- **Cotton Wool Spots**: Localized ischaemic infarcts of the retinal nerve fibre layer.\n"
            "- **Neovascularisation**: Pathological new fragile vessels prone to spontaneous vitreous bleeding."
        )

    elif any(k in q for k in ["hba1c", "blood sugar", "diet", "prevent", "control"]):
        return (
            "**Metabolic Control & Progression Risk:**\n\n"
            "Rigorous landmark clinical trials (DCCT and UKPDS) demonstrate that sustained HbA1c reduction "
            "substantially delays the onset and retards the progression of diabetic retinopathy. "
            "Every 1% absolute reduction in HbA1c reduces microvascular complication risk by approximately 35%. "
            "Strict blood pressure control (<130/80 mmHg) and lipid management are equally essential."
        )

    else:
        return (
            f"Regarding **{details['full_name']}**: The current triage status is **{details['triage_label']}** "
            f"with an action target of **{details['triage_timeline']}**.\n\n"
            "You can ask me about:\n"
            "- Referral timelines and specialist review\n"
            "- Grad-CAM attention heatmap interpretation\n"
            "- Specific retinal lesions (microaneurysms, exudates, hemorrhages)\n"
            "- Metabolic guidelines (HbA1c, blood pressure control)"
        )
