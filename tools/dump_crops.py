#!/usr/bin/env python3
"""Dump corner crops + Python ground-truth for the browser-port parity workflow.

The browser detector uses its own (deterministic) HOG, so we retrain the logistic
regression in *that* feature space. Crop generation + augmentation stay in Python
(cv2), exactly as train_camera_clf.py does, so the browser model trains on the
same pixels the Python model did. This script emits two JSON files:

  tools/parity/train_crops.json   augmented 32x32 crops + label + image-group id
                                  (consumed by tools/train_js_clf.mjs to retrain)
  tools/parity/reference.json     per-image, per-corner raw crop + Python cam_prob
                                  + Python verdict (consumed by the JS parity check)

    python tools/dump_crops.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from pipeline import preprocess, segment, locate, camera_clf as CC
from detect_meta_glasses import run as run_pipeline

POS = "ray_ban_frame"   # label 1
NEG = "normal_frame"    # label 0
OUT_DIR = os.path.join("tools", "parity")


def files(folder):
    return sorted(f for f in glob.glob(f"{folder}/*")
                  if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))


def crops_for_image(path, cfg):
    """(crop, side) per corner ROI — mirrors train_camera_clf.crops_for_image."""
    fr = preprocess.preprocess(path, cfg)
    seg = segment.segment(fr.gray_eq, cfg)
    loc = locate.locate(fr.gray_eq, seg, cfg)
    return [(CC.corner_crop(fr, roi), roi.side) for roi in loc.rois]


def augment(crop):
    """Identical to train_camera_clf.augment (label-preserving crop variants)."""
    import cv2
    out = [crop]
    h, w = crop.shape
    for scale in (0.82, 1.18):
        out.append(np.clip(crop.astype(np.float32) * scale, 0, 255).astype(np.uint8))
    for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        out.append(cv2.warpAffine(crop, M, (w, h), borderMode=cv2.BORDER_REPLICATE))
    for ang in (-6, 6):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        out.append(cv2.warpAffine(crop, M, (w, h), borderMode=cv2.BORDER_REPLICATE))
    return out


def main() -> int:
    cfg = Config()
    os.makedirs(OUT_DIR, exist_ok=True)

    train = []        # {pixels:[1024], label, group}
    reference = []    # {name, label, verdict, corners:[{side, crop, cam_prob}]}

    group = 0
    for folder, label in ((POS, 1), (NEG, 0)):
        for path in files(folder):
            name = os.path.basename(path)

            # training crops (augmented), grouped by image for leave-one-out CV
            for crop, _side in crops_for_image(path, cfg):
                for a in augment(crop):
                    train.append({"pixels": a.reshape(-1).astype(int).tolist(),
                                  "label": label, "group": group})

            # reference: raw crops + Python's own verdict/cam_prob for this image
            frames, seg, loc, feats, verdict = run_pipeline(path, cfg)
            corners = []
            for roi in loc.rois:
                crop = CC.corner_crop(frames, roi)
                corners.append({
                    "side": roi.side,
                    "crop": crop.reshape(-1).astype(int).tolist(),
                    "cam_prob": feats[roi.side].cam_prob,
                })
            reference.append({"name": name, "label": label,
                              "verdict": verdict.verdict, "corners": corners})
            group += 1

    with open(os.path.join(OUT_DIR, "train_crops.json"), "w") as fh:
        json.dump(train, fh)
    with open(os.path.join(OUT_DIR, "reference.json"), "w") as fh:
        json.dump(reference, fh)

    npos = sum(1 for r in reference if r["label"] == 1)
    print(f"images: {len(reference)} ({npos} camera, {len(reference)-npos} normal)")
    print(f"train crops (augmented): {len(train)}")
    print(f"wrote {OUT_DIR}/train_crops.json and reference.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
