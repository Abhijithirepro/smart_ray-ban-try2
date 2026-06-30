/* Detector configuration — mirrors the trimmed config.py (classifier-only).
 * All spatial params are relative to frame width / lens box (scale-invariant
 * after the canonical resize). Works under <script> (window.DET.config) and Node.
 */
(function (root, factory) {
  var mod = factory();
  if (typeof module !== 'undefined' && module.exports) { module.exports = mod; }
  if (root) { root.DET = root.DET || {}; root.DET.config = mod; }
})(typeof self !== 'undefined' ? self : (typeof window !== 'undefined' ? window : null),
function () {
  'use strict';
  return {
    // preprocess
    target_width: 1000,
    clahe_clip: 2.0,
    clahe_tile: 8,
    // segment (frame bbox)
    seg_close_ksize: 7,
    seg_open_ksize: 3,
    seg_min_area_frac: 0.10,
    seg_max_area_frac: 0.97,
    seg_min_aspect: 1.4,
    seg_max_aspect: 4.5,
    seg_isolated_max: 0.80,
    canny_lo: 50,
    canny_hi: 150,
    // locate (top-outer corner ROI the classifier scores)
    roi_lens_overlap: 0.15,
    roi_y_below: 0.15,
    roi_min_w_frac: 0.10,
    // learned classifier
    cam_clf_thresh: 0.50
  };
});
