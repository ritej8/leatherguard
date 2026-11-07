"""Modules 3 + 4: wrinkle detector (YOLO-OBB) -> crops -> UNet++ segmentation -> measurements.

Ported from the training notebook (`run_full_inference`, `seg_predict_fn`,
`obb_to_padded_crop`, `measure_defect`). torch / ultralytics / smp are imported lazily
so the pure functions can be unit-tested without them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .grading import assign_grade

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CROP_PADDING_RATIO = 0.4   # same values as the dataset construction
MIN_CROP_SIZE = 96
SEG_SIZE = 384             # segmentation was trained and evaluated at 384x384
SEG_THRESHOLD = 0.45       # best value of the threshold sweep in the notebook
DET_CONF = 0.15
DET_IMGSZ = 960            # detector training size


# ----------------------------------------------------------------- pure helpers
def obb_to_padded_crop(coords_norm, w, h, padding_ratio=CROP_PADDING_RATIO, min_size=MIN_CROP_SIZE):
    """Enlarge an oriented box (8 normalised coords) into an axis-aligned padded crop."""
    pts = np.asarray(coords_norm, dtype=np.float64).reshape(4, 2) * np.array([w, h])
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    bw, bh = x2 - x1, y2 - y1
    x1, y1 = x1 - bw * padding_ratio, y1 - bh * padding_ratio
    x2, y2 = x2 + bw * padding_ratio, y2 + bh * padding_ratio
    if (x2 - x1) < min_size:
        cx = (x1 + x2) / 2
        x1, x2 = cx - min_size / 2, cx + min_size / 2
    if (y2 - y1) < min_size:
        cy = (y1 + y2) / 2
        y1, y2 = cy - min_size / 2, cy + min_size / 2
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    return x1, y1, x2, y2


def postprocess_mask(mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    """Morphological close/open, then drop connected blobs smaller than min_area px."""
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(clean, [c], -1, 255, -1)
    return clean


def measure_defect(mask: np.ndarray) -> Optional[dict]:
    """Geometry of the largest connected defect in a 0/255 mask (None if empty)."""
    binary = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    _, (rw, rh), angle = cv2.minAreaRect(largest)
    hull_area = cv2.contourArea(cv2.convexHull(largest))
    length, width = max(rw, rh), min(rw, rh)
    return dict(
        area_px=area,
        length_px=length,
        width_px=width,
        perimeter_px=perimeter,
        circularity=4 * np.pi * area / (perimeter ** 2 + 1e-6),
        solidity=area / (hull_area + 1e-6),
        aspect_ratio=length / (width + 1e-6),
        orientation_deg=angle,
    )


# ----------------------------------------------------------------- models
def get_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_detector(path):
    from ultralytics import YOLO

    model = YOLO(str(path))
    if getattr(model, "task", None) != "obb":
        raise ValueError(f"Detector task is '{model.task}', expected 'obb' (oriented boxes).")
    return model


def load_segmenter(path, device: str = "cpu"):
    """U-Net++ / ResNet34, single-channel output. Weights come from the file, so no
    ImageNet download is needed (encoder_weights=None)."""
    import segmentation_models_pytorch as smp
    import torch

    model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.eval().to(device)


def predict_tta(model, tensor):
    """Average of the identity, horizontal-flip and vertical-flip predictions."""
    import torch

    with torch.no_grad():
        p1 = torch.sigmoid(model(tensor))
        p2 = torch.sigmoid(model(torch.flip(tensor, dims=[3]))).flip(dims=[3])
        p3 = torch.sigmoid(model(torch.flip(tensor, dims=[2]))).flip(dims=[2])
    return (p1 + p2 + p3) / 3


def segment_crop(model, crop_rgb: np.ndarray, device="cpu", img_size=SEG_SIZE,
                 threshold=SEG_THRESHOLD, min_area=50) -> np.ndarray:
    """Segment one RGB crop. Returns a 0/255 uint8 mask at the crop's own size."""
    import torch

    orig_h, orig_w = crop_rgb.shape[:2]
    img = cv2.resize(crop_rgb, (img_size, img_size)).astype(np.float32) / 255.0
    img = (img - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    tensor = torch.from_numpy(img.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device)
    probs = predict_tta(model, tensor)
    pred = (probs > threshold).float().squeeze().cpu().numpy().astype(np.uint8) * 255
    pred = postprocess_mask(pred, min_area)
    return cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


# ----------------------------------------------------------------- full pipeline
@dataclass
class PipelineResult:
    full_mask: np.ndarray                     # (H, W) uint8, 0/255, full image size
    polygons: List[np.ndarray] = field(default_factory=list)   # detected OBBs, (4, 2) px
    crop_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    measurements: Optional[dict] = None       # largest defect
    percentage: float = 0.0                   # largest defect area / image area * 100
    total_percentage: float = 0.0             # all defect pixels / image area * 100
    grade: str = "A"
    decision: str = "Accept"


def run_pipeline(image: Image.Image, detector, segmenter, device="cpu", det_conf=DET_CONF,
                 det_imgsz=DET_IMGSZ, seg_size=SEG_SIZE, seg_threshold=SEG_THRESHOLD) -> PipelineResult:
    pil = image.convert("RGB")
    rgb = np.asarray(pil)
    h, w = rgb.shape[:2]

    result = detector.predict(source=pil, conf=det_conf, imgsz=det_imgsz, verbose=False)[0]
    polygons: List[np.ndarray] = []
    if getattr(result, "obb", None) is not None and len(result.obb) > 0:
        polygons = [p.reshape(4, 2) for p in result.obb.xyxyxyxy.cpu().numpy()]

    full_mask = np.zeros((h, w), np.uint8)
    crop_boxes = []
    for poly in polygons:
        coords_norm = (poly / np.array([w, h])).reshape(-1)
        x1, y1, x2, y2 = obb_to_padded_crop(coords_norm, w, h)
        if x2 <= x1 or y2 <= y1:
            continue
        crop_boxes.append((x1, y1, x2, y2))
        mask = segment_crop(segmenter, rgb[y1:y2, x1:x2], device, seg_size, seg_threshold)
        full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], mask)

    image_area = h * w
    measurements = measure_defect(full_mask)
    percentage = measurements["area_px"] / image_area * 100 if measurements else 0.0
    total_percentage = float((full_mask > 127).sum()) / image_area * 100
    grade, decision = assign_grade(percentage) if measurements else ("A", "Accept")
    return PipelineResult(full_mask, polygons, crop_boxes, measurements, percentage,
                          total_percentage, grade, decision)


def render_overlay(image: Image.Image, res: PipelineResult, max_side: int = 1280, alpha: float = 0.45) -> Image.Image:
    """Red mask + yellow detected boxes, downscaled for display."""
    rgb = np.asarray(image.convert("RGB")).copy()
    mask = res.full_mask
    polys = [p.copy() for p in res.polygons]
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        size = (int(w * scale), int(h * scale))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        polys = [p * scale for p in polys]
    on = mask > 127
    rgb[on] = ((1 - alpha) * rgb[on] + alpha * np.array([255, 0, 0])).astype(np.uint8)
    thickness = max(2, rgb.shape[1] // 400)
    cv2.polylines(rgb, [p.astype(np.int32) for p in polys], True, (255, 255, 0), thickness)
    return Image.fromarray(rgb)
