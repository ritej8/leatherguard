"""LeatherGuard - Streamlit demo of the 3-module pipeline.

Module 2  classifier   : photo -> P(wrinkle) -> OK / DEFECT / REVIEW
Module 3  detector     : YOLO-OBB boxes around wrinkles
Module 4  segmenter    : U-Net++ mask inside each box -> defect area -> grade A/B/C
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from leatherguard.classifier import load_classifier, predict_proba
from leatherguard.inference import DEFECT_CLASS, decide, display_copy, prepare_image
from leatherguard.pipeline import (
    DET_CONF, SEG_THRESHOLD, get_device, load_detector, load_segmenter,
    render_overlay, run_pipeline,
)

MODELS = Path(__file__).parent / "models"
CLASSIFIER_PATH = MODELS / "classifier.pt"
DETECTOR_PATH = MODELS / "detector.pt"
SEGMENTER_PATH = MODELS / "segmenter.pt"
THRESHOLD_PATH = MODELS / "decision_threshold.json"

st.set_page_config(page_title="LeatherGuard", page_icon="🧵", layout="wide")
DEVICE = get_device()


@st.cache_resource(show_spinner="Loading classifier...")
def get_classifier(path: str):
    return load_classifier(path, DEVICE)


@st.cache_resource(show_spinner="Loading detector...")
def get_detector(path: str):
    return load_detector(path)


@st.cache_resource(show_spinner="Loading segmenter...")
def get_segmenter(path: str):
    return load_segmenter(path, DEVICE)


def default_threshold() -> float:
    """Decision threshold tuned on the validation set by the training notebook."""
    try:
        return float(json.loads(THRESHOLD_PATH.read_text())["threshold"])
    except Exception:
        return 0.5


st.title("🧵 LeatherGuard")
st.caption("Leather wrinkle inspection - research prototype, not validated for production use.")

if not CLASSIFIER_PATH.exists():
    st.error("Missing `models/classifier.pt`")
    st.markdown("Export your weights and put them in `models/` (see the README, *Exporting the models*).")
    st.stop()

classifier, arch = get_classifier(str(CLASSIFIER_PATH))
localisation_available = DETECTOR_PATH.exists() and SEGMENTER_PATH.exists()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Classification")
    threshold = st.slider(
        "Defect threshold", 0.01, 0.99, min(max(default_threshold(), 0.01), 0.99), 0.01,
        help="DEFECT when P(wrinkle) >= threshold. The default is the value tuned on the "
             "validation set (aiming at high recall). Lower = fewer missed defects, more false alarms.",
    )
    review_band = st.slider(
        "Manual-review band", 0.0, 0.30, 0.05, 0.05,
        help="P(wrinkle) within this distance of the threshold -> REVIEW (send to a human).",
    )
    st.caption(f"Classifier: {arch} - running on {DEVICE.upper()}")

    st.header("Localisation + grading")
    always_localise = st.checkbox(
        "Also localise when the verdict is OK", value=False, disabled=not localisation_available,
        help="By default localisation only runs on images flagged DEFECT or REVIEW.",
    )
    det_conf = st.slider("Detector confidence", 0.05, 0.90, DET_CONF, 0.05, disabled=not localisation_available)
    seg_thr = st.slider("Segmentation threshold", 0.20, 0.80, SEG_THRESHOLD, 0.05, disabled=not localisation_available)
    if not localisation_available:
        st.info("Add `detector.pt` and `segmenter.pt` to `models/` to enable localisation and grading.")

detector = get_detector(str(DETECTOR_PATH)) if localisation_available else None
segmenter = get_segmenter(str(SEGMENTER_PATH)) if localisation_available else None

# ---------------------------------------------------------------- input
files = st.file_uploader("Upload leather photo(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if not files:
    st.info("Upload one or more photos to start.")
    st.stop()

BADGE = {"OK": st.success, "DEFECT": st.error, "REVIEW": st.warning}
TEXT = {
    "OK": "✅ OK - no wrinkle detected",
    "DEFECT": "❌ DEFECT - wrinkle detected",
    "REVIEW": "⚠️ REVIEW - too close to call, check manually",
}

rows = []
for file in files:
    try:
        image = prepare_image(file)  # full resolution on purpose
    except Exception as exc:
        st.warning(f"Could not read `{file.name}`: {exc}")
        continue

    with st.spinner(f"Analysing {file.name}..."):
        probs = predict_proba(classifier, image, DEVICE)
        verdict = decide(probs, DEFECT_CLASS, threshold, review_band)
        res = None
        if localisation_available and (verdict.label != "OK" or always_localise):
            res = run_pipeline(image, detector, segmenter, DEVICE, det_conf, seg_threshold=seg_thr)

    left, right = st.columns([3, 2])
    with left:
        if res is not None and res.polygons:
            st.image(render_overlay(image, res), caption=f"{file.name} - {len(res.polygons)} region(s) localised")
        else:
            st.image(display_copy(image), caption=file.name)
    with right:
        BADGE[verdict.label](TEXT[verdict.label])
        st.metric(f"P({DEFECT_CLASS})", f"{verdict.defect_prob:.1%}")
        st.progress(min(max(verdict.defect_prob, 0.0), 1.0))
        if res is not None:
            if res.measurements is None:
                st.warning("No wrinkle region could be localised.")
            else:
                st.metric("Defect area (largest region)", f"{res.percentage:.2f}% of the image")
                st.write(f"**Grade {res.grade}** - {res.decision}")
                st.caption(
                    f"Length {res.measurements['length_px']:.0f}px, width {res.measurements['width_px']:.0f}px. "
                    "Grade grid: <2% A, 2-5% B, >5% C, >20% REVIEW (starting point, not a standard)."
                )
    st.divider()

    rows.append(
        {
            "file": file.name,
            "verdict": verdict.label,
            f"p_{DEFECT_CLASS}": round(verdict.defect_prob, 4),
            "regions": len(res.polygons) if res is not None else None,
            "defect_pct": round(res.percentage, 3) if res is not None and res.measurements else None,
            "grade": res.grade if res is not None and res.measurements else None,
        }
    )

if rows:
    table = pd.DataFrame(rows)
    st.subheader("Summary")
    st.dataframe(table, hide_index=True)
    st.write(" | ".join(f"{k}: {v}" for k, v in table["verdict"].value_counts().to_dict().items()))
    st.download_button("Download results (CSV)", table.to_csv(index=False).encode("utf-8"),
                       file_name="leatherguard_results.csv", mime="text/csv")
