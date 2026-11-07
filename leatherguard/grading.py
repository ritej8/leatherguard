"""Severity grading from the defect area (pure Python, no dependencies).

The grid comes from the training notebook and is described there as a *starting
point to be tuned with the industrial partner*. It is not a validated standard.
"""
from typing import Tuple

GRADE_A_MAX = 2.0              # % of the image area
GRADE_B_MAX = 5.0
MAX_PLAUSIBLE_PERCENTAGE = 20.0  # above this the segmentation is suspected to be wrong


def assign_grade(percentage: float) -> Tuple[str, str]:
    """Return (grade, decision) for a defect covering `percentage` % of the image."""
    if percentage < 0:
        raise ValueError("percentage must be >= 0")
    if percentage > MAX_PLAUSIBLE_PERCENTAGE:
        return "REVIEW", "Manual check required"
    if percentage < GRADE_A_MAX:
        return "A", "Accept"
    if percentage <= GRADE_B_MAX:
        return "B", "Rework"
    return "C", "Reject"
