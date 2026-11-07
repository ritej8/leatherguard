"""Image loading and decision logic shared by the whole app (no heavy imports)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from PIL import Image, ImageOps

DEFECT_CLASS = "wrinkle"


@dataclass(frozen=True)
class Verdict:
    label: str          # "DEFECT", "OK" or "REVIEW"
    defect_prob: float  # probability of the defect class
    top_class: str      # most likely class according to the model
    top_prob: float


def prepare_image(file_or_path, max_side: Optional[int] = None) -> Image.Image:
    """Open an image, honour EXIF orientation (phone photos) and convert to RGB.

    EXIF handling matters: training images were read with OpenCV, which applies the
    EXIF rotation, so inference must do the same. By default the image keeps its full
    resolution: the detector/segmenter were trained on full-resolution crops, so
    downscaling first would blur the crops. Use `display_copy` for the screen.
    """
    image = Image.open(file_or_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max_side:
        image.thumbnail((max_side, max_side))
    return image


def display_copy(image: Image.Image, max_side: int = 1280) -> Image.Image:
    """Smaller copy for display (a 24 MP image can crash the browser tab)."""
    copy = image.copy()
    copy.thumbnail((max_side, max_side))
    return copy


def decide(
    probs: Dict[str, float],
    defect_class: str = DEFECT_CLASS,
    threshold: float = 0.5,
    review_band: float = 0.0,
) -> Verdict:
    """Turn class probabilities into an inspection verdict.

    - |defect_prob - threshold| < band    -> REVIEW (too close to call, send to a human)
    - defect_prob >= threshold            -> DEFECT
    - defect_prob <  threshold            -> OK

    Lowering `threshold` increases recall on defects (fewer missed wrinkles) at
    the price of more false alarms.
    """
    if defect_class not in probs:
        raise KeyError(
            f"Defect class '{defect_class}' not found in model classes {sorted(probs)}"
        )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if review_band < 0:
        raise ValueError("review_band must be >= 0")

    defect_prob = float(probs[defect_class])
    top_class = max(probs, key=probs.get)
    top_prob = float(probs[top_class])

    if abs(defect_prob - threshold) < review_band:
        label = "REVIEW"
    elif defect_prob >= threshold:
        label = "DEFECT"
    else:
        label = "OK"
    return Verdict(label, defect_prob, top_class, top_prob)
