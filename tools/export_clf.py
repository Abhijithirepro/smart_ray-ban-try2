#!/usr/bin/env python3
"""Export the trained corner classifier (.npz) to JSON for the browser port.

The frontend-only detector can't read numpy's .npz (a zip of .npy); JSON is
trivial to fetch and parse in the browser. The exported weights are consumed by
static/detector/camera_clf.js, whose prob() mirrors CameraClf.prob exactly:

    sigmoid( ((feat - mu) / sd) @ w + b )

    python tools/export_clf.py            # writes static/camera_clf.json
"""
from __future__ import annotations

import json
import os
import sys

# allow running from the project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from pipeline.camera_clf import CameraClf


def main() -> int:
    cfg = Config()
    if not os.path.exists(cfg.cam_clf_path):
        print(f"error: no weights at {cfg.cam_clf_path}", file=sys.stderr)
        return 2
    clf = CameraClf.load(cfg.cam_clf_path)
    out = {
        "w": clf.w.astype(float).tolist(),
        "b": float(clf.b),
        "mu": clf.mu.astype(float).tolist(),
        "sd": clf.sd.astype(float).tolist(),
        "thresh": float(cfg.cam_clf_thresh),
    }
    dst = os.path.join("static", "camera_clf.json")
    with open(dst, "w") as fh:
        json.dump(out, fh)
    print(f"wrote {dst}  (dim={len(out['w'])}, b={out['b']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
