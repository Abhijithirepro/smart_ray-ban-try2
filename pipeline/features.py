"""Stage 4 - per-corner camera probability from the learned classifier.

The verdict is decided entirely by the learned corner classifier
(pipeline/camera_clf). For each top-outer corner ROI placed by locate, we take
its orientation-canonical crop and read off P(camera). No hand-tuned circle /
blob / darkness / glint measurements are used any more - that distinction is
structural/semantic, so we let the model make it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from config import Config
from pipeline.locate import Located
from pipeline.segment import Segment


@dataclass
class CornerFeatures:
    side: str
    cam_prob: float | None = None        # learned P(camera) for this corner crop


_CLF_CACHE = {}


def _load_clf(path: str):
    """Lazily load (and cache) the corner camera classifier; None if absent."""
    if path not in _CLF_CACHE:
        from pipeline import camera_clf
        _CLF_CACHE[path] = (camera_clf.CameraClf.load(path)
                            if os.path.exists(path) else None)
    return _CLF_CACHE[path]


def extract(frames, seg: Segment, loc: Located, cfg: Config):
    """Return {'L': CornerFeatures, 'R': CornerFeatures} with learned P(camera)."""
    clf = _load_clf(cfg.cam_clf_path)
    result = {}
    for roi in loc.rois:
        f = CornerFeatures(side=roi.side)
        if clf is not None:
            from pipeline import camera_clf
            f.cam_prob = round(
                clf.prob_from_crop(camera_clf.corner_crop(frames, roi)), 3)
        result[roi.side] = f
    return result
