#!/usr/bin/env python3
"""Dump each reference image as raw RGBA bytes for the Node parity harness.

Feeding Node the *same* cv2-decoded pixels (alpha flattened onto white, as
preprocess does) isolates pipeline-logic parity from browser-decoder differences:
any crop/verdict mismatch in Node is a port bug, not a JPEG-decoder discrepancy.

Writes tools/parity/raw/<i>.bin (RGBA, row-major) + tools/parity/raw/manifest.json
in the SAME order as reference.json (index i lines up).

    python tools/dump_raw.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.preprocess import load_image, _flatten_alpha

POS = "ray_ban_frame"
NEG = "normal_frame"
RAW_DIR = os.path.join("tools", "parity", "raw")


def files(folder):
    return sorted(f for f in glob.glob(f"{folder}/*")
                  if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))


def main() -> int:
    os.makedirs(RAW_DIR, exist_ok=True)
    manifest = []
    i = 0
    for folder, label in ((POS, 1), (NEG, 0)):
        for path in files(folder):
            bgr = _flatten_alpha(load_image(path))     # same as preprocess input
            h, w = bgr.shape[:2]
            # BGR -> RGBA (alpha 255), matching a white-backed browser canvas
            rgba = np.dstack([bgr[:, :, 2], bgr[:, :, 1], bgr[:, :, 0],
                              np.full((h, w), 255, np.uint8)])
            rgba.tofile(os.path.join(RAW_DIR, f"{i}.bin"))
            manifest.append({"i": i, "name": os.path.basename(path),
                             "w": int(w), "h": int(h), "label": label})
            i += 1
    with open(os.path.join(RAW_DIR, "manifest.json"), "w") as fh:
        json.dump(manifest, fh)
    print(f"wrote {i} raw RGBA images to {RAW_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
