"""
test_integration.py — End-to-end real model integration test.
Run from repo root: python test_integration.py
"""
import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, ".")
sys.path.insert(0, "src")

from pathlib import Path

# 1. Checkpoint detection
print("[1] Checking checkpoint availability ...")
from app.predictor import get_checkpoint_path, is_real_model_available
chk = get_checkpoint_path()
assert chk is not None, "No checkpoint found!"
print(f"    Checkpoint : {chk}")
print(f"    Size       : {chk.stat().st_size / 1e6:.1f} MB")

# 2. Model loading
print("[2] Loading real PyTorch model ...")
from app.predictor import get_real_model
model, device = get_real_model()
assert model is not None, "Model failed to load!"
print(f"    Device     : {device}")
print(f"    Status     : loaded OK")

# 3. Real inference on a test image
print("[3] Running real inference on test image ...")
test_dir = Path("data/split/test")
img_path = None
for cls in ["Mild", "Moderate", "No_DR", "Severe", "Proliferative_DR"]:
    candidates = list((test_dir / cls).glob("*.png"))
    if candidates:
        img_path = candidates[0]
        break

assert img_path is not None, "No test image found in data/split/test/"
print(f"    Image      : {img_path.name} (class: {img_path.parent.name})")

from app.predictor import predict_image
with open(img_path, "rb") as f:
    img_bytes = f.read()

result = predict_image(img_bytes, force_mock=False, patient_id="PT-TEST-001")

print(f"    Predicted  : {result['predicted_class']}")
print(f"    Confidence : {result['confidence_pct']}")
print(f"    Engine     : {result['engine_mode']}")
print(f"    Probs sum  : {sum(result['probabilities']):.4f}")
print(f"    GradCAM    : {len(result['gradcam_image_bytes'])} bytes")

# Validate output structure matches what the UI expects
required_keys = [
    "patient_id", "eye", "predicted_class", "class_index",
    "confidence", "confidence_pct", "probabilities", "class_names",
    "details", "triage_label", "triage_timeline", "triage_category",
    "summary", "original_image_bytes", "gradcam_image_bytes", "engine_mode"
]
for key in required_keys:
    assert key in result, f"Missing key in result: {key}"
assert result["engine_mode"] == "PyTorch Deep CNN (EfficientNet-B0 Staged Fine-Tuned)", \
    f"Unexpected engine mode: {result['engine_mode']}"
assert len(result["probabilities"]) == 5
assert 0.0 < result["confidence"] <= 1.0
assert abs(sum(result["probabilities"]) - 1.0) < 0.01, "Probabilities don't sum to 1.0!"

# 4. Test patient summary PDF generation still works
print("[4] Testing patient PDF generation ...")
from app.pdf_report import generate_patient_summary_pdf
pdf_bytes = generate_patient_summary_pdf(result)
assert len(pdf_bytes) > 1000, "Patient PDF too small!"
print(f"    Patient PDF: {len(pdf_bytes) / 1024:.1f} KB")

# 5. Test clinical PDF still works
print("[5] Testing clinical PDF generation ...")
from app.pdf_report import generate_clinical_pdf_report
cpdf_bytes = generate_clinical_pdf_report(result)
assert len(cpdf_bytes) > 1000, "Clinical PDF too small!"
print(f"    Clinical PDF: {len(cpdf_bytes) / 1024:.1f} KB")

print()
print("=" * 55)
print("  ALL INTEGRATION TESTS PASSED")
print("=" * 55)
