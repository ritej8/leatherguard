"""Tests of the pure helpers (need numpy + opencv only, no torch, no weights)."""
import numpy as np
import pytest

from leatherguard.classifier import detect_arch
from leatherguard.grading import assign_grade
from leatherguard.pipeline import measure_defect, obb_to_padded_crop, postprocess_mask


# ---- architecture auto-detection (key names follow torchvision)
def test_detect_efficientnet():
    assert detect_arch(["features.0.0.weight", "classifier.1.weight", "classifier.1.bias"]) == "EfficientNetV2-S"


def test_detect_resnet():
    assert detect_arch(["conv1.weight", "layer1.0.conv1.weight", "fc.weight"]) == "ResNet50"


def test_detect_mobilenet():
    keys = ["features.0.0.weight", "classifier.0.weight", "classifier.3.weight"]
    assert detect_arch(keys) == "MobileNetV3-Large"


def test_detect_unknown_raises():
    with pytest.raises(ValueError):
        detect_arch(["encoder.conv1.weight", "decoder.blocks.0.weight"])


# ---- grading grid
@pytest.mark.parametrize(
    "pct, expected",
    [(0.0, "A"), (1.99, "A"), (2.0, "B"), (5.0, "B"), (5.01, "C"), (20.0, "C"), (20.01, "REVIEW")],
)
def test_grade_boundaries(pct, expected):
    assert assign_grade(pct)[0] == expected


def test_grade_decisions():
    assert assign_grade(1)[1] == "Accept"
    assert assign_grade(3)[1] == "Rework"
    assert assign_grade(10)[1] == "Reject"


# ---- crop geometry
def test_crop_padding_and_bounds():
    # box (100..200, 100..200) in a 1000x1000 image, padding 0.4 -> 60..240
    coords = np.array([100, 100, 200, 100, 200, 200, 100, 200]) / 1000
    assert obb_to_padded_crop(coords, 1000, 1000) == (60, 60, 240, 240)


def test_crop_min_size_and_clipping_at_border():
    coords = np.array([0, 0, 10, 0, 10, 10, 0, 10]) / 1000
    x1, y1, x2, y2 = obb_to_padded_crop(coords, 1000, 1000)
    assert x1 == 0 and y1 == 0           # clipped at the image border
    assert x2 - x1 >= 48 and y2 - y1 >= 48  # enlarged towards min_size, never negative


# ---- mask post-processing / measurement
def test_postprocess_removes_tiny_blobs_and_keeps_big_ones():
    mask = np.zeros((200, 200), np.uint8)
    mask[10:13, 10:13] = 255              # 9 px blob -> removed
    mask[100:160, 100:140] = 255          # 2400 px blob -> kept
    out = postprocess_mask(mask, min_area=50)
    assert out[11, 11] == 0 and out[130, 120] == 255


def test_measure_defect():
    mask = np.zeros((300, 300), np.uint8)
    mask[100:120, 50:250] = 255           # 200 x 20 bar
    m = measure_defect(mask)
    assert 3500 < m["area_px"] < 4100
    assert m["aspect_ratio"] > 8
    assert measure_defect(np.zeros((50, 50), np.uint8)) is None
