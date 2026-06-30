/* Deterministic HOG descriptor for the corner-camera classifier (browser port).
 *
 * cv2.HOGDescriptor (Gaussian window + trilinear interpolation) is not exposed
 * by OpenCV.js and is fiddly to reproduce bit-for-bit, so the browser model is
 * RETRAINED on this descriptor instead (see tools/train_js_clf.mjs). The only
 * requirement is therefore self-consistency: train and inference must call this
 * exact function. The layout mirrors the Python descriptor's geometry:
 *
 *   32x32 input -> 8x8 cells (4x4 = 16 cells) -> 9 unsigned-orientation bins
 *   -> 2x2 non-overlapping 16x16 blocks (each = 2x2 cells) -> L2-Hys per block
 *   -> 4 blocks x 36 = 144-dim descriptor.
 *
 * Works under both <script> (attaches to window.DET) and Node (module.exports).
 */
(function (root, factory) {
  var mod = factory();
  if (typeof module !== 'undefined' && module.exports) { module.exports = mod; }
  if (root) { root.DET = root.DET || {}; root.DET.hog = mod; }
})(typeof self !== 'undefined' ? self : (typeof window !== 'undefined' ? window : null),
function () {
  'use strict';

  var SIDE = 32;        // crop is SIDE x SIDE, row-major
  var CELL = 8;         // cell size in px
  var NCELL = SIDE / CELL;   // 4 cells per axis
  var NBINS = 9;        // unsigned orientation bins over [0,180)
  var BIN_W = 180 / NBINS;   // 20 degrees
  var BLK = 2;          // block = BLK x BLK cells (16x16 px, stride 16 => 2x2 blocks)
  var EPS = 1e-6;
  var L2HYS = 0.2;

  /**
   * Compute the 144-dim HOG descriptor of a 32x32 grayscale crop.
   * @param {Uint8Array|Array|Float32Array} px  length-1024 row-major pixels (0..255)
   * @returns {Float64Array} length-144 descriptor
   */
  function hogFeat(px) {
    // per-cell histograms: [NCELL][NCELL][NBINS]
    var hist = new Float64Array(NCELL * NCELL * NBINS);

    var x, y;
    for (y = 0; y < SIDE; y += 1) {
      for (x = 0; x < SIDE; x += 1) {
        // central difference with replicate border (clamp index)
        var xm = x > 0 ? x - 1 : 0;
        var xp = x < SIDE - 1 ? x + 1 : SIDE - 1;
        var ym = y > 0 ? y - 1 : 0;
        var yp = y < SIDE - 1 ? y + 1 : SIDE - 1;
        var gx = px[y * SIDE + xp] - px[y * SIDE + xm];
        var gy = px[yp * SIDE + x] - px[ym * SIDE + x];
        var mag = Math.sqrt(gx * gx + gy * gy);
        if (mag <= 0) { continue; }

        // unsigned orientation in [0,180)
        var ang = Math.atan2(gy, gx) * (180 / Math.PI);
        if (ang < 0) { ang += 180; }
        if (ang >= 180) { ang -= 180; }

        // linear interpolation between the two nearest bins (orientation only)
        var bf = ang / BIN_W - 0.5;
        var b0 = Math.floor(bf);
        var frac = bf - b0;
        var bin0 = ((b0 % NBINS) + NBINS) % NBINS;
        var bin1 = (bin0 + 1) % NBINS;

        var cx = (x / CELL) | 0;
        var cy = (y / CELL) | 0;
        var base = (cy * NCELL + cx) * NBINS;
        hist[base + bin0] += mag * (1 - frac);
        hist[base + bin1] += mag * frac;
      }
    }

    // 2x2 non-overlapping blocks, each = 2x2 cells = 36 dims, L2-Hys normalised
    var out = new Float64Array(NCELL * NCELL * NBINS); // 16*9 = 144
    var o = 0;
    var bi, bj, ci, cj, k;
    for (bj = 0; bj < NCELL; bj += BLK) {
      for (bi = 0; bi < NCELL; bi += BLK) {
        // gather block vector
        var blk = [];
        for (cj = 0; cj < BLK; cj += 1) {
          for (ci = 0; ci < BLK; ci += 1) {
            var cbase = (((bj + cj) * NCELL) + (bi + ci)) * NBINS;
            for (k = 0; k < NBINS; k += 1) { blk.push(hist[cbase + k]); }
          }
        }
        // L2 normalise
        var s = 0, i;
        for (i = 0; i < blk.length; i += 1) { s += blk[i] * blk[i]; }
        var n = Math.sqrt(s + EPS);
        for (i = 0; i < blk.length; i += 1) { blk[i] /= n; }
        // Hys clip then renormalise
        s = 0;
        for (i = 0; i < blk.length; i += 1) {
          if (blk[i] > L2HYS) { blk[i] = L2HYS; }
          s += blk[i] * blk[i];
        }
        n = Math.sqrt(s + EPS);
        for (i = 0; i < blk.length; i += 1) { out[o] = blk[i] / n; o += 1; }
      }
    }
    return out;
  }

  return { hogFeat: hogFeat, DIM: NCELL * NCELL * NBINS };
});
