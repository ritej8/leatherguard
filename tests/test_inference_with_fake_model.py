"""Tests classify()/detect()/prepare_image() with a fake model that mimics the
Ultralytics result API, so they run without torch or weights."""
import io

import numpy as np
from PIL import Image

from leatherguard.inference import classify, detect, prepare_image


class _T:  # minimal tensor stand-in
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def tolist(self):
        return self.a.tolist()


class FakeClsModel:
    def predict(self, source, imgsz, verbose):
        assert isinstance(source, Image.Image) and source.mode == "RGB"
        probs = type("P", (), {"data": _T([0.2, 0.8])})()
        return [type("R", (), {"probs": probs, "names": {0: "normal", 1: "wrinkle"}})()]


class FakeDetModel:
    def predict(self, source, imgsz, conf, verbose):
        boxes = type("B", (), {"xyxy": _T([[10, 10, 50, 40]]), "conf": _T([0.77])})()
        return [type("R", (), {"boxes": boxes})()]


def test_classify_maps_names_to_probs():
    out = classify(FakeClsModel(), Image.new("RGB", (64, 64)))
    assert out == {"normal": 0.2, "wrinkle": 0.8}


def test_detect_returns_boxes_and_annotated_image():
    img = Image.new("RGB", (100, 100), "white")
    annotated, boxes = detect(FakeDetModel(), img)
    assert len(boxes) == 1 and boxes[0]["confidence"] == 0.77
    assert annotated.size == img.size and annotated.getpixel((10, 10)) != (255, 255, 255)


def test_prepare_image_downscales_and_applies_exif_rotation():
    img = Image.new("RGB", (4000, 2000), "red")
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90 degrees CW to display"
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    buf.seek(0)
    out = prepare_image(buf, max_side=1000)
    assert max(out.size) == 1000
    assert out.height > out.width  # rotated to portrait
