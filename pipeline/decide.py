"""Stage 5 - verdict from the learned per-corner camera probability.

A Ray-Ban Meta carries a circular camera module in BOTH top-outer corners, so we
fire META iff the learned classifier reports P(camera) >= cam_clf_thresh in BOTH
corner ROIs. A domain gate first abstains when no isolated glasses frame was
found (e.g. a face photo), where the "corners" are meaningless.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import Config


@dataclass
class Verdict:
    verdict: str          # "META" | "NORMAL"
    prob_left: float | None
    prob_right: float | None
    fired_corner: str | None
    reason: str


def decide(features: dict, cfg: Config, seg=None, shape=None) -> Verdict:
    fL, fR = features["L"], features["R"]
    pL, pR = fL.cam_prob, fR.cam_prob

    # Domain gate: we only claim META when an isolated glasses frame was found.
    # If segmentation collapsed (e.g. a face photo, not a clean glasses shot),
    # the "corners" are meaningless and an eye can mimic a camera, so abstain.
    if seg is not None and shape is not None and not seg.isolated(cfg, shape):
        return Verdict(verdict="NORMAL", prob_left=pL, prob_right=pR,
                       fired_corner=None,
                       reason="no isolated glasses frame found (out of operating "
                              "condition: needs a clean photo of just the glasses)")

    if pL is None or pR is None:
        raise RuntimeError(
            "camera classifier unavailable: expected weights at "
            f"{cfg.cam_clf_path} (train with train_camera_clf.py)")

    has_L = pL >= cfg.cam_clf_thresh
    has_R = pR >= cfg.cam_clf_thresh
    cue = f"P(cam) L={pL:.2f} R={pR:.2f}"

    is_meta = has_L and has_R
    if is_meta:
        reason = f"camera in both corners [{cue}]"
        fired = "L+R"
    elif has_L or has_R:
        only = "L" if has_L else "R"
        reason = f"camera only in corner {only}, need both [{cue}]"
        fired = None
    else:
        reason = f"no camera in either corner [{cue}]"
        fired = None

    return Verdict(verdict="META" if is_meta else "NORMAL",
                   prob_left=pL, prob_right=pR,
                   fired_corner=fired, reason=reason)
