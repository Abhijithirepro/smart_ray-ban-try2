/* Stage 5 - verdict from the learned per-corner camera probability.
 * Port of pipeline/decide.py: META iff P(camera) >= thresh in BOTH corners,
 * with a domain gate that abstains when no isolated glasses frame was found.
 */
(function (root, factory) {
  var mod = factory();
  if (typeof module !== 'undefined' && module.exports) { module.exports = mod; }
  if (root) { root.DET = root.DET || {}; root.DET.decide = mod; }
})(typeof self !== 'undefined' ? self : (typeof window !== 'undefined' ? window : null),
function () {
  'use strict';

  function f2(p) { return p.toFixed(2); }

  /**
   * @param {number} pL per-corner P(camera), left
   * @param {number} pR per-corner P(camera), right
   * @param {boolean} isolated  seg.isolated()
   * @param {object} cfg
   * @returns {{verdict, prob_left, prob_right, fired_corner, reason}}
   */
  function decide(pL, pR, isolated, cfg) {
    if (!isolated) {
      return {
        verdict: 'NORMAL', prob_left: pL, prob_right: pR, fired_corner: null,
        reason: 'no isolated glasses frame found (out of operating condition: '
          + 'needs a clean photo of just the glasses)'
      };
    }
    var thr = cfg.cam_clf_thresh;
    var hasL = pL >= thr, hasR = pR >= thr;
    var cue = 'P(cam) L=' + f2(pL) + ' R=' + f2(pR);
    var isMeta = hasL && hasR;
    var reason, fired;
    if (isMeta) { reason = 'camera in both corners [' + cue + ']'; fired = 'L+R'; }
    else if (hasL || hasR) {
      var only = hasL ? 'L' : 'R';
      reason = 'camera only in corner ' + only + ', need both [' + cue + ']';
      fired = null;
    } else { reason = 'no camera in either corner [' + cue + ']'; fired = null; }

    return {
      verdict: isMeta ? 'META' : 'NORMAL',
      prob_left: pL, prob_right: pR, fired_corner: fired, reason: reason
    };
  }

  return { decide: decide };
});
