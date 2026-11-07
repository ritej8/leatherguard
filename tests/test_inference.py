"""Unit tests for the decision logic (no model or GPU needed)."""
import pytest

from leatherguard.inference import decide


PROBS_DEFECT = {"normal": 0.10, "wrinkle": 0.90}
PROBS_OK = {"normal": 0.95, "wrinkle": 0.05}
PROBS_BORDER = {"normal": 0.48, "wrinkle": 0.52}


def test_defect():
    v = decide(PROBS_DEFECT)
    assert v.label == "DEFECT" and v.top_class == "wrinkle"


def test_ok():
    assert decide(PROBS_OK).label == "OK"


def test_threshold_is_inclusive():
    assert decide({"normal": 0.5, "wrinkle": 0.5}, threshold=0.5).label == "DEFECT"


def test_lower_threshold_catches_more_defects():
    probs = {"normal": 0.7, "wrinkle": 0.3}
    assert decide(probs, threshold=0.5).label == "OK"
    assert decide(probs, threshold=0.25).label == "DEFECT"


def test_review_band():
    assert decide(PROBS_BORDER, review_band=0.1).label == "REVIEW"
    assert decide(PROBS_BORDER, review_band=0.0).label == "DEFECT"
    assert decide(PROBS_DEFECT, review_band=0.1).label == "DEFECT"


def test_unknown_defect_class():
    with pytest.raises(KeyError):
        decide({"a": 0.5, "b": 0.5})


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        decide(PROBS_OK, threshold=threshold)
