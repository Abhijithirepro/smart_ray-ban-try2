/* Corner-camera classifier: HOG -> standardise -> linear logit -> sigmoid.
 *
 * Mirrors pipeline/camera_clf.py. prob() is the direct port of CameraClf.prob;
 * the weights live in static/camera_clf.json (retrained in the JS-HOG space by
 * tools/train_js_clf.mjs). cornerCrop() reproduces camera_clf.corner_crop and
 * needs OpenCV.js (cv) for the ROI resize/flip; the rest is pure JS.
 *
 * Works under <script> (window.DET.clf) and Node (module.exports).
 */
(function (root, factory) {
  var mod = factory(root);
  if (typeof module !== 'undefined' && module.exports) { module.exports = mod; }
  if (root) { root.DET = root.DET || {}; root.DET.clf = mod; }
})(typeof self !== 'undefined' ? self : (typeof window !== 'undefined' ? window : null),
function (root) {
  'use strict';

  var hog = (root && root.DET && root.DET.hog)
    ? root.DET.hog
    : (typeof require !== 'undefined' ? require('./hog.js') : null);

  var CROP = 32;

  /** sigmoid(((feat - mu)/sd) . w + b) — exact port of CameraClf.prob. */
  function prob(feat, model) {
    var z = model.b, i;
    for (i = 0; i < feat.length; i += 1) {
      z += ((feat[i] - model.mu[i]) / model.sd[i]) * model.w[i];
    }
    return 1 / (1 + Math.exp(-z));
  }

  /** P(camera) for a 32x32 crop (length-1024 px array). */
  function probFromCrop(px, model) {
    return prob(hog.hogFeat(px), model);
  }

  /**
   * Fixed-size, orientation-canonical grayscale crop of a corner ROI.
   * Port of camera_clf.corner_crop: gray ROI -> 32x32 (INTER_AREA) -> R flipped.
   * @param {cv.Mat} grayMat  full-frame raw grayscale (CV_8UC1)
   * @param {{x,y,w,h,side}} roi
   * @param {object} cv  the OpenCV.js module
   * @returns {Uint8Array} length-1024 row-major crop pixels
   */
  function cornerCrop(grayMat, roi, cv) {
    if (roi.w <= 0 || roi.h <= 0) { return new Uint8Array(CROP * CROP); }
    var rect = new cv.Rect(roi.x, roi.y, roi.w, roi.h);
    var sub = grayMat.roi(rect);
    var dst = new cv.Mat();
    cv.resize(sub, dst, new cv.Size(CROP, CROP), 0, 0, cv.INTER_AREA);
    if (roi.side === 'R') {
      var f = new cv.Mat();
      cv.flip(dst, f, 1);
      dst.delete();
      dst = f;
    }
    var out = new Uint8Array(dst.data); // copy CV_8UC1 bytes
    sub.delete();
    dst.delete();
    return out;
  }

  return { prob: prob, probFromCrop: probFromCrop, cornerCrop: cornerCrop, CROP: CROP };
});
