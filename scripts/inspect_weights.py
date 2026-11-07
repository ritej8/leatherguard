"""Check that the three exported weight files load, and print what they are.

    python scripts/inspect_weights.py
    python scripts/inspect_weights.py --classifier "best_model (2).pt" \
        --detector "best (4).pt" --segmenter "unetpp_reference_seed42 (3).pt"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", default=ROOT / "models" / "classifier.pt")
    ap.add_argument("--detector", default=ROOT / "models" / "detector.pt")
    ap.add_argument("--segmenter", default=ROOT / "models" / "segmenter.pt")
    args = ap.parse_args()
    failures = 0

    print("== Classifier ==")
    try:
        import torch
        from leatherguard.classifier import CLASSES, INPUT_SIZE, load_classifier

        model, arch = load_classifier(args.classifier)
        out = model(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE))
        print(f"OK  architecture={arch}  outputs={tuple(out.shape)}  classes={CLASSES}")
    except Exception as exc:
        failures += 1
        print("FAIL", type(exc).__name__, exc)

    print("== Detector ==")
    try:
        from leatherguard.pipeline import load_detector

        det = load_detector(args.detector)
        print(f"OK  task={det.task}  classes={det.names}")
    except Exception as exc:
        failures += 1
        print("FAIL", type(exc).__name__, exc)

    print("== Segmenter ==")
    try:
        import torch
        from leatherguard.pipeline import SEG_SIZE, load_segmenter

        seg = load_segmenter(args.segmenter)
        with torch.no_grad():
            out = seg(torch.zeros(1, 3, SEG_SIZE, SEG_SIZE))
        print(f"OK  U-Net++/ResNet34  output={tuple(out.shape)}")
    except Exception as exc:
        failures += 1
        print("FAIL", type(exc).__name__, exc)

    print("\nAll good." if not failures else f"\n{failures} file(s) failed - see messages above.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
