/* Stage 2 - frame bbox + filled frame mask. Port of pipeline/segment.py.
 *
 * Primary: inverse-Otsu -> close/open -> largest external contour -> bbox
 * (rejected by an area/aspect gate). Fallbacks: Canny contour, then centre crop.
 */
(function (root, factory) {
  var mod = factory();
  if (typeof module !== 'undefined' && module.exports) { module.exports = mod; }
  if (root) { root.DET = root.DET || {}; root.DET.segment = mod; }
})(typeof self !== 'undefined' ? self : (typeof window !== 'undefined' ? window : null),
function () {
  'use strict';

  function odd(k) { return k | 1; }

  /** Index of the largest-area contour in a MatVector, or -1. */
  function largestIdx(contours, cv) {
    var best = -1, bestA = -1, i;
    for (i = 0; i < contours.size(); i += 1) {
      var a = cv.contourArea(contours.get(i));
      if (a > bestA) { bestA = a; best = i; }
    }
    return best;
  }

  function bboxOk(b, W, H, cfg) {
    var areaFrac = (b.width * b.height) / (W * H);
    if (!(cfg.seg_min_area_frac < areaFrac && areaFrac < cfg.seg_max_area_frac)) { return false; }
    var aspect = b.height ? b.width / b.height : 0;
    return cfg.seg_min_aspect <= aspect && aspect <= cfg.seg_max_aspect;
  }

  /** Filled mask (uint8 {0,255}) from one contour of a MatVector. */
  function filledMask(contours, idx, H, W, cv) {
    var mask = cv.Mat.zeros(H, W, cv.CV_8UC1);
    var white = new cv.Scalar(255, 255, 255, 255);
    cv.drawContours(mask, contours, idx, white, -1); // -1 = FILLED
    return mask;
  }

  function rectKernel(k, cv) {
    return cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(k, k));
  }

  /**
   * @param {cv.Mat} grayEq CV_8UC1
   * @returns {{bbox:[x,y,w,h], frameMask:cv.Mat, method:string, valid:boolean,
   *            isolated:Function, free:Function}}
   */
  function segment(grayEq, cfg, cv) {
    var H = grayEq.rows, W = grayEq.cols;

    function makeResult(bbox, mask, method, valid) {
      return {
        bbox: bbox,
        frameMask: mask,
        method: method,
        valid: valid,
        isolated: function () {
          if (!this.valid || this.method === 'center-crop') { return false; }
          var areaFrac = (this.bbox[2] * this.bbox[3]) / (W * H);
          return areaFrac <= cfg.seg_isolated_max;
        },
        free: function () { this.frameMask.delete(); }
      };
    }

    // ---- primary: inverse Otsu ----
    var th = new cv.Mat();
    cv.threshold(grayEq, th, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
    var kClose = rectKernel(odd(cfg.seg_close_ksize), cv);
    var kOpen = rectKernel(odd(cfg.seg_open_ksize), cv);
    var anchor = new cv.Point(-1, -1);
    cv.morphologyEx(th, th, cv.MORPH_CLOSE, kClose, anchor, 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
    cv.morphologyEx(th, th, cv.MORPH_OPEN, kOpen, anchor, 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
    kClose.delete(); kOpen.delete();

    var result = null;
    var contours = new cv.MatVector();
    var hier = new cv.Mat();
    cv.findContours(th, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    var idx = largestIdx(contours, cv);
    if (idx >= 0) {
      var b = cv.boundingRect(contours.get(idx));
      if (bboxOk(b, W, H, cfg)) {
        result = makeResult([b.x, b.y, b.width, b.height],
          filledMask(contours, idx, H, W, cv), 'otsu', true);
      }
    }
    contours.delete(); hier.delete(); th.delete();
    if (result) { return result; }

    // ---- fallback: Canny edges ----
    var edges = new cv.Mat();
    cv.Canny(grayEq, edges, cfg.canny_lo, cfg.canny_hi);
    var k3 = rectKernel(3, cv);
    cv.dilate(edges, edges, k3, new cv.Point(-1, -1), 2);
    k3.delete();
    var c2 = new cv.MatVector();
    var h2 = new cv.Mat();
    cv.findContours(edges, c2, h2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    var idx2 = largestIdx(c2, cv);
    if (idx2 >= 0) {
      var b2 = cv.boundingRect(c2.get(idx2));
      if (bboxOk(b2, W, H, cfg)) {
        result = makeResult([b2.x, b2.y, b2.width, b2.height],
          filledMask(c2, idx2, H, W, cv), 'canny', true);
      }
    }
    c2.delete(); h2.delete(); edges.delete();
    if (result) { return result; }

    // ---- last resort: central 90% crop ----
    var x = Math.floor(0.05 * W), y = Math.floor(0.05 * H);
    var w = Math.floor(0.90 * W), hh = Math.floor(0.90 * H);
    var mask = cv.Mat.zeros(H, W, cv.CV_8UC1);
    var roi = mask.roi(new cv.Rect(x, y, w, hh));
    roi.setTo(new cv.Scalar(255, 255, 255, 255));
    roi.delete();
    return makeResult([x, y, w, hh], mask, 'center-crop', false);
  }

  return { segment: segment };
});
