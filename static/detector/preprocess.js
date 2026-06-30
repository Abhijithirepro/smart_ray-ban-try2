/* Stage 1 - canonical resize + grayscale variants. Port of pipeline/preprocess.py.
 *
 * Produces the working images the rest of the pipeline shares:
 *   color   : resized colour (kept for the overlay)
 *   gray    : raw grayscale (used by corner_crop)
 *   grayEq  : CLAHE-equalised gray (used by segment / locate)
 * The now-dead gray_blur (Hough input) is omitted.
 *
 * Input is an RGBA cv.Mat (CV_8UC4) at original size — exactly what the browser
 * canvas yields. cvtColor RGBA->GRAY reproduces cv2's BGR->GRAY luminance.
 *
 * Caller owns srcRgba; this returns Mats the caller must delete via free().
 */
(function (root, factory) {
  var mod = factory();
  if (typeof module !== 'undefined' && module.exports) { module.exports = mod; }
  if (root) { root.DET = root.DET || {}; root.DET.preprocess = mod; }
})(typeof self !== 'undefined' ? self : (typeof window !== 'undefined' ? window : null),
function () {
  'use strict';

  /**
   * @param {cv.Mat} srcRgba  CV_8UC4 original image
   * @param {object} cfg
   * @param {object} cv  OpenCV.js module
   * @returns {{color:cv.Mat, gray:cv.Mat, grayEq:cv.Mat, scale:number,
   *            origShape:[number,number], free:Function}}
   */
  function preprocess(srcRgba, cfg, cv) {
    var origH = srcRgba.rows;
    var origW = srcRgba.cols;
    var scale = cfg.target_width / origW;
    var newW = cfg.target_width;
    var newH = Math.max(1, Math.round(origH * scale));

    var color = new cv.Mat();
    var interp = scale < 1.0 ? cv.INTER_AREA : cv.INTER_CUBIC;
    cv.resize(srcRgba, color, new cv.Size(newW, newH), 0, 0, interp);

    var gray = new cv.Mat();
    cv.cvtColor(color, gray, cv.COLOR_RGBA2GRAY);

    var grayEq = new cv.Mat();
    var clahe = new cv.CLAHE(cfg.clahe_clip, new cv.Size(cfg.clahe_tile, cfg.clahe_tile));
    clahe.apply(gray, grayEq);
    clahe.delete();

    return {
      color: color,
      gray: gray,
      grayEq: grayEq,
      scale: scale,
      origShape: [origH, origW],
      free: function () { color.delete(); gray.delete(); grayEq.delete(); }
    };
  }

  return { preprocess: preprocess };
});
