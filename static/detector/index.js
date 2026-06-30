/* Detector orchestrator — runs the whole in-browser pipeline and returns the
 * same payload shape the Flask /api/detect endpoint produced (viz.features_to_dict),
 * so static/app.js renders it unchanged.
 *
 *   preprocess -> segment -> locate -> corner_crop -> HOG -> logistic -> decide
 *
 * detectMat() takes a prebuilt RGBA cv.Mat (the caller owns/builds it from a
 * canvas in the browser, or from raw bytes in the Node parity harness).
 */
(function (root, factory) {
  var mod = factory(root);
  if (typeof module !== 'undefined' && module.exports) { module.exports = mod; }
  if (root) { root.DET = root.DET || {}; root.DET.detectMat = mod.detectMat; }
})(typeof self !== 'undefined' ? self : (typeof window !== 'undefined' ? window : null),
function (root) {
  'use strict';

  function dep(name) {
    if (root && root.DET && root.DET[name]) { return root.DET[name]; }
    if (typeof require !== 'undefined') { return require('./' + name + '.js'); }
    throw new Error('missing detector dependency: ' + name);
  }

  function round3(p) { return Math.round(p * 1000) / 1000; }

  /**
   * @param {cv.Mat} rgbaMat CV_8UC4 source image
   * @param {object} cv OpenCV.js module
   * @param {object} model {w,b,mu,sd,thresh}
   * @param {object} [cfg]  defaults to DET.config
   * @returns {object} payload
   */
  function detectMat(rgbaMat, cv, model, cfg) {
    var config = cfg || dep('config');
    var preprocess = dep('preprocess').preprocess;
    var segment = dep('segment').segment;
    var locate = dep('locate').locate;
    var clf = dep('clf');
    var decide = dep('decide').decide;

    var pre = preprocess(rgbaMat, config, cv);
    var seg = segment(pre.grayEq, config, cv);
    var loc = locate(pre.grayEq, seg, config, cv);

    var per = { L: { cam_prob: null }, R: { cam_prob: null } };
    var i;
    for (i = 0; i < loc.rois.length; i += 1) {
      var roi = loc.rois[i];
      var crop = clf.cornerCrop(pre.gray, roi, cv);
      per[roi.side] = { cam_prob: round3(clf.probFromCrop(crop, model)) };
    }

    var verdict = decide(per.L.cam_prob, per.R.cam_prob, seg.isolated(), config);

    var payload = {
      verdict: verdict.verdict,
      prob_left: verdict.prob_left,
      prob_right: verdict.prob_right,
      fired_corner: verdict.fired_corner,
      reason: verdict.reason,
      r_lens: Math.round(loc.rLens * 100) / 100,
      bbox: seg.bbox.slice(),
      segment_method: seg.method,
      locate_method: loc.method,
      lens_centers: {
        L: [Math.round(loc.lensLeft.cx * 10) / 10, Math.round(loc.lensLeft.cy * 10) / 10],
        R: [Math.round(loc.lensRight.cx * 10) / 10, Math.round(loc.lensRight.cy * 10) / 10]
      },
      per_corner: per,
      threshold: config.cam_clf_thresh,
      // geometry for the optional in-browser overlay (app.js draws it)
      geom: { bbox: seg.bbox.slice(), rois: loc.rois, lensL: loc.lensLeft, lensR: loc.lensRight,
              colorW: pre.color.cols, colorH: pre.color.rows }
    };

    seg.free();
    pre.free();
    return payload;
  }

  return { detectMat: detectMat };
});
