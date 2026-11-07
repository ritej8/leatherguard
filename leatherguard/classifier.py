"""Module 2 - image classifier (normal / wrinkle), torchvision state_dict.

The training notebook compares three architectures and saves each winner as a
plain `state_dict` called best_model.pt, so the architecture is NOT stored in the
file. `detect_arch` recovers it from the layer names.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 260                      # Resize((260, 260)) at training time
CLASSES = ("normal", "wrinkle")       # index order used at training time

# architecture -> name of its final Linear layer in the state_dict
HEAD_KEY = {
    "ResNet50": "fc.weight",
    "MobileNetV3-Large": "classifier.3.weight",
    "EfficientNetV2-S": "classifier.1.weight",
}


def detect_arch(keys: Iterable[str]) -> str:
    keys = set(keys)
    if "fc.weight" in keys:
        return "ResNet50"
    if "classifier.3.weight" in keys:
        return "MobileNetV3-Large"
    if "classifier.1.weight" in keys:
        return "EfficientNetV2-S"
    raise ValueError(
        "Unrecognised classifier weights (expected EfficientNetV2-S, ResNet50 or "
        "MobileNetV3-Large from torchvision)."
    )


def build_classifier(arch: str, n_classes: int):
    import torch.nn as nn
    import torchvision.models as models

    if arch == "EfficientNetV2-S":
        m = models.efficientnet_v2_s(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    elif arch == "ResNet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif arch == "MobileNetV3-Large":
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
    else:
        raise ValueError(arch)
    return m


def load_classifier(path, device: str = "cpu") -> Tuple[object, str]:
    """Load the weights, auto-detecting the architecture. Returns (model, arch_name)."""
    import torch

    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items()}  # DataParallel

    arch = detect_arch(state.keys())
    n_classes = state[HEAD_KEY[arch]].shape[0]
    if n_classes != len(CLASSES):
        raise ValueError(f"The model has {n_classes} outputs, expected {len(CLASSES)} {CLASSES}")

    model = build_classifier(arch, n_classes)
    model.load_state_dict(state)
    return model.eval().to(device), arch


def predict_proba(model, image, device: str = "cpu") -> Dict[str, float]:
    """Return {class_name: probability} for a PIL image."""
    import torch
    from torchvision import transforms

    tf = transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    x = tf(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().tolist()
    return dict(zip(CLASSES, probs))
