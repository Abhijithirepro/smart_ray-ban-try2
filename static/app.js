/* MODULE SCAN — front-end controller (ES5, vanilla).
   Runs the ENTIRE detection pipeline in the browser (OpenCV.js + the detector
   modules in static/detector/), with no server. Handles upload (click +
   drag/drop) and live camera, renders the verdict, confidence meter, per-corner
   P(camera) and an annotated overlay drawn on a canvas. */
(function () {
  'use strict';

  /* current input mode: 'camera' (default) or 'upload' */
  var mode = 'camera';

  /* the learned classifier weights ({w,b,mu,sd,thresh}); fetched on load */
  var model = null;
  /* OpenCV.js runtime readiness */
  var cvReady = false;

  /** True once both OpenCV.js and the model JSON have loaded. */
  function detectReady() { return cvReady && model !== null; }

  /** Run cb once the OpenCV.js wasm runtime is initialised (race-safe). */
  function whenCvReady(cb) {
    if (window.cv && window.cv.Mat) { cb(); return; }
    if (window.cv) { window.cv.onRuntimeInitialized = cb; }
    var iv = window.setInterval(function () {
      if (window.cv && window.cv.Mat) { window.clearInterval(iv); cb(); }
    }, 50);
  }

  /**
   * Grab an element by id (null-safe helper).
   * @param {string} id
   * @returns {HTMLElement|null}
   */
  function el(id) {
    return document.getElementById(id);
  }

  /**
   * Show a status line, optionally flagged as an error or busy state.
   * @param {string} msg
   * @param {string} mode  '' | 'error' | 'busy'
   */
  function setStatus(msg, mode) {
    var s = el('status');
    if (!s) { return; }
    s.textContent = msg;
    s.className = 'status' + (mode ? ' is-' + mode : '');
  }

  /**
   * Format a 0..1 number to two decimals.
   * @param {number} n
   * @returns {string}
   */
  function fmt(n) {
    return (Math.round(n * 100) / 100).toFixed(2);
  }

  /**
   * Read the per-corner camera probabilities (0..1) the verdict is based on.
   * `has` is false when no classifier is loaded (pipeline fell back to Hough).
   * @param {Object} payload
   * @returns {{l:number, r:number, has:boolean}}
   */
  function camProbs(payload) {
    var pf = payload.per_corner || {};
    var L = pf.L || {}, R = pf.R || {};
    var ok = function (v) { return v !== null && v !== undefined; };
    return { l: L.cam_prob, r: R.cam_prob, has: ok(L.cam_prob) && ok(R.cam_prob) };
  }

  /**
   * Format a 0..1 number as a whole percentage.
   * @param {number} n
   * @returns {string}
   */
  function pct(n) {
    return Math.round(n * 100) + '%';
  }

  /**
   * Paint one corner card with its camera probability and pass/fail state.
   * @param {string} scoreId
   * @param {string} stateId
   * @param {string} cardId
   * @param {number} prob
   * @param {number} thr
   * @param {boolean} hasProb
   */
  function renderCorner(scoreId, stateId, cardId, prob, thr, hasProb) {
    var sc = el(scoreId), st = el(stateId), card = el(cardId);
    if (!sc || !st || !card) { return; }
    if (!hasProb) {
      sc.textContent = 'n/a'; st.textContent = '—'; card.className = 'corner';
      return;
    }
    sc.textContent = pct(prob);
    if (prob >= thr) { st.textContent = 'CAMERA'; card.className = 'corner is-cam'; }
    else { st.textContent = 'no camera'; card.className = 'corner is-nocam'; }
  }

  /**
   * Build the plain-English explanation of why the verdict came out this way.
   * @param {Object} payload
   * @param {{l:number, r:number, has:boolean}} cp
   * @param {number} thr
   * @returns {string}
   */
  function whyText(payload, cp, thr) {
    var isMeta = payload.verdict === 'META';
    /* domain-gate / Hough-fallback cases: the server reason is the truth */
    if (!cp.has) { return payload.reason || ''; }
    if (!isMeta && payload.reason &&
        payload.reason.indexOf('isolated') !== -1) {
      return payload.reason;
    }
    var L = pct(cp.l), R = pct(cp.r), T = pct(thr);
    if (isMeta) {
      return 'A camera module was recognised in BOTH top-outer corners — ' +
        'left ' + L + ' and right ' + R + ' confidence (need ≥ ' + T +
        '). Ray-Ban Meta glasses carry a camera in each corner, so both ' +
        'corners passing is what flags this as smart glasses.';
    }
    var miss = (cp.l < thr && cp.r < thr) ? 'either corner' :
               (cp.l < thr ? 'the left corner' : 'the right corner');
    return 'No camera module in ' + miss + ' (left ' + L + ', right ' + R +
      '; need ≥ ' + T + '). Both corners must contain a camera for ' +
      'smart glasses, so this reads as normal eyeglasses.';
  }

  /**
   * Paint the full result panel from an API payload.
   * @param {Object} payload
   */
  function renderResult(payload) {
    var result = el('result');
    var isMeta = payload.verdict === 'META';
    var threshold = payload.threshold || 0.5;
    var cp = camProbs(payload);
    /* the meter shows the BINDING corner: both must pass, so the weaker one
       is what the verdict hinges on */
    var conf = cp.has ? Math.min(cp.l, cp.r) : 0;

    el('overlay').src = payload.overlay || '';
    el('verdict-text').textContent = isMeta ? 'SMART GLASSES' : 'NORMAL GLASSES';
    el('verdict-reason').textContent = isMeta
      ? 'camera recognised in both corners'
      : 'not smart glasses';

    el('conf-val').textContent = fmt(conf);
    renderCorner('score-l', 'state-l', 'corner-l', cp.l, threshold, cp.has);
    renderCorner('score-r', 'state-r', 'corner-r', cp.r, threshold, cp.has);
    el('why-text').textContent = whyText(payload, cp, threshold);

    /* the loop reruns automatically in camera mode, so hide the manual reset */
    var again = el('again');
    if (again) { again.hidden = (mode === 'camera'); }

    result.className = 'result ' + (isMeta ? 'is-meta' : 'is-normal');
    result.hidden = false;
    document.body.className = 'has-result';

    /* run the scan reveal, then fill the meter once layout is settled */
    void result.offsetWidth;
    result.className += ' is-scanning';
    window.setTimeout(function () {
      el('conf-fill').style.width = (conf * 100) + '%';
      el('conf-mark').style.left = (threshold * 100) + '%';
    }, 120);
    window.setTimeout(function () {
      result.className = 'result ' + (isMeta ? 'is-meta' : 'is-normal');
    }, 1100);

    setStatus('scan complete · ' + payload.segment_method + ' segmentation', '');
    result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  /**
   * Upload an image (file picker upload OR a captured webcam frame) to the
   * detector and render the response.
   * @param {Blob|File} file       image File, or a canvas Blob (webcam frame)
   * @param {Function} [onComplete] called after the request settles (ok or error)
   */
  function scan(file, onComplete) {
    var done = function (ok) { if (typeof onComplete === 'function') { onComplete(ok); } };
    if (!file) { done(false); return; }
    if (file.type && file.type.indexOf('image/') !== 0) {
      setStatus('that is not an image file', 'error');
      done(false);
      return;
    }
    if (!detectReady()) {
      setStatus('detector still loading — one moment …', 'busy');
      done(false);
      return;
    }
    setStatus('scanning corners for a camera module', 'busy');

    var url = URL.createObjectURL(file);
    var img = new Image();
    img.onload = function () {
      var payload;
      try {
        payload = runDetect(img);
      } catch (e) {
        URL.revokeObjectURL(url);
        setStatus('detector error: ' + (e && e.message ? e.message : e), 'error');
        done(false);
        return;
      }
      URL.revokeObjectURL(url);
      renderResult(payload);
      done(true);
    };
    img.onerror = function () {
      URL.revokeObjectURL(url);
      setStatus('could not read that image', 'error');
      done(false);
    };
    img.src = url;
  }

  /**
   * Run the full in-browser pipeline on a loaded image and return the payload
   * (same shape the old /api/detect endpoint produced), with an overlay drawn.
   * @param {HTMLImageElement} img
   * @returns {Object}
   */
  function runDetect(img) {
    var cv = window.cv;
    var w = img.naturalWidth || img.width;
    var h = img.naturalHeight || img.height;
    var c = document.createElement('canvas');
    c.width = w; c.height = h;
    var ctx = c.getContext('2d');
    /* No white fill: the Python pipeline reads images with cv2.imread(IMREAD_COLOR),
       which drops the alpha channel. A transparent canvas reads opaque pixels back
       identically to cv2 and avoids white-compositing skew on RGBA inputs. Live
       webcam frames are fully opaque, so this path is exact for camera mode. */
    ctx.drawImage(img, 0, 0, w, h);
    var id = ctx.getImageData(0, 0, w, h);
    var src = cv.matFromImageData(id);
    var payload;
    try {
      payload = window.DET.detectMat(src, cv, model, window.DET.config);
    } finally {
      src.delete();
    }
    payload.overlay = buildOverlay(img, payload);
    return payload;
  }

  /**
   * Draw the annotated overlay (frame bbox, lens boxes, corner ROIs + P(camera),
   * verdict banner) on a canvas at canonical scale and return a data URI. This
   * reproduces pipeline/viz.annotate in the browser.
   * @param {HTMLImageElement} img
   * @param {Object} payload
   * @returns {string} PNG data URI
   */
  function buildOverlay(img, payload) {
    var g = payload.geom || {};
    var W = g.colorW || 1000, H = g.colorH || 1000;
    var c = document.createElement('canvas');
    c.width = W; c.height = H;
    var ctx = c.getContext('2d');
    ctx.drawImage(img, 0, 0, W, H);

    function rect(x, y, w, h, color, lw) {
      ctx.strokeStyle = color; ctx.lineWidth = lw || 2;
      ctx.strokeRect(x, y, w, h);
    }
    var bb = g.bbox || payload.bbox;
    if (bb) { rect(bb[0], bb[1], bb[2], bb[3], '#2d7dff', 2); }
    [g.lensL, g.lensR].forEach(function (ln) {
      if (!ln) { return; }
      rect(ln.cx - ln.hw, ln.cy - ln.hh, ln.hw * 2, ln.hh * 2, '#27e08a', 1);
    });
    (g.rois || []).forEach(function (roi) {
      rect(roi.x, roi.y, roi.w, roi.h, '#ffd23f', 2);
      var p = (payload.per_corner[roi.side] || {}).cam_prob;
      ctx.fillStyle = '#ffd23f';
      ctx.font = '14px "Space Mono", monospace';
      ctx.fillText(roi.side + (p == null ? '' : '  Pcam=' + fmt(p)),
        roi.x, Math.max(12, roi.y - 4));
    });
    /* header banner */
    var isMeta = payload.verdict === 'META';
    ctx.fillStyle = 'rgba(0,0,0,0.78)';
    ctx.fillRect(0, 0, W, 28);
    ctx.fillStyle = isMeta ? '#ff4d4d' : '#27e08a';
    ctx.font = '700 16px "Space Mono", monospace';
    var pL = payload.prob_left, pR = payload.prob_right;
    var banner = payload.verdict + (pL == null ? '' :
      '  Pcam L=' + fmt(pL) + ' R=' + fmt(pR));
    ctx.fillText(banner, 6, 20);
    return c.toDataURL('image/png');
  }

  /* ======================================================================
     LIVE CAMERA MODE
     The how-it-works clip plays in the intro column (a separate place), while
     the camera scene shows the live feed straight away:
     enter → live preview + START (clip auto-plays alongside)
           → START → [5s countdown → grab → analyse]∞
     ====================================================================== */

  var CYCLE_SECONDS = 5;       // live-preview countdown before each capture
  var RESULT_HOLD_MS = 3000;   // how long a verdict stays before the next cycle

  var stream = null;           // the active MediaStream (kept across cycles)
  var timers = [];             // pending setTimeout handles
  var ticker = null;           // the countdown setInterval handle
  var spokenOnce = false;      // the instructions narration plays only once per load

  /**
   * Speak the instructions once. Browsers block autoplay-with-sound until a
   * user gesture, so if play() is rejected we retry on the first tap/click.
   */
  function playInstructions() {
    if (spokenOnce) { return; }
    var audio = el('intro-audio');
    if (!audio) { return; }
    spokenOnce = true;
    var fired = false;
    var go = function () {
      if (fired) { return; }
      fired = true;
      try { audio.currentTime = 0; } catch (e) {}
      audio.play();
    };
    var p = audio.play();
    if (p && typeof p.catch === 'function') {
      p.catch(function () {
        /* autoplay blocked — arm a one-shot gesture listener to start it */
        var unlock = function () {
          document.removeEventListener('pointerdown', unlock);
          document.removeEventListener('keydown', unlock);
          go();
        };
        document.addEventListener('pointerdown', unlock);
        document.addEventListener('keydown', unlock);
      });
    }
  }

  /** Clear every pending timer/interval so a mode switch fully stops the loop. */
  function clearTimers() {
    var i;
    for (i = 0; i < timers.length; i += 1) { window.clearTimeout(timers[i]); }
    timers = [];
    if (ticker !== null) { window.clearInterval(ticker); ticker = null; }
  }

  /** setTimeout that registers its handle for clearTimers(). */
  function later(fn, ms) {
    var id = window.setTimeout(fn, ms);
    timers.push(id);
    return id;
  }

  /** Toggle the [hidden] attribute on an element by id (null-safe). */
  function show(id, visible) {
    var n = el(id);
    if (n) { n.hidden = !visible; }
  }

  /**
   * Enter live-camera mode: request the webcam, then show the live preview with
   * START straight away in the camera scene while the how-it-works clip plays in
   * the intro column. The clip never occupies the camera scene.
   */
  function enterCamera() {
    mode = 'camera';
    clearTimers();
    show('camera-mode', true);
    show('upload-mode', false);
    show('live-wrap', false);
    show('cam-controls', false);
    show('cam-instruction', true);
    el('cam-instruction').textContent =
      'Initialising camera … allow access when prompted.';

    playIntro();          // clip lives in the intro column, independent of the feed
    playInstructions();   // spoken instructions, once per page load

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus('this browser has no camera access; use Upload Image', 'error');
      return;
    }
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(function (s) {
        if (mode !== 'camera') { s.getTracks().forEach(function (t) { t.stop(); }); return; }
        stream = s;
        readyToScan();
      })
      .catch(function () {
        setStatus('camera access denied — switch to Upload Image instead', 'error');
      });
  }

  /**
   * Show the live self-view preview plus the START / REPLAY controls, ready for
   * the user to begin scanning whenever they like.
   */
  function readyToScan() {
    if (mode !== 'camera' || !stream) { return; }
    show('cam-instruction', false);
    var live = el('live');
    live.srcObject = stream;
    el('countdown').hidden = true;
    el('cam-state').textContent = '';
    show('live-wrap', true);
    show('cam-controls', true);
    setStatus('watch the how-it-works clip, then press START TEST to scan', '');
  }

  /**
   * Play the instructional clip from the start, in its own panel in the intro
   * column. Used on entry and on REPLAY; failures (blocked autoplay) are silent.
   */
  function playIntro() {
    show('intro-clip', true);
    var intro = el('intro');
    if (!intro) { return; }
    intro.currentTime = 0;
    var p = intro.play();
    if (p && typeof p.catch === 'function') { p.catch(function () {}); }
  }

  /** Leave camera mode: stop the loop, release the webcam, reset the UI. */
  function leaveCamera() {
    clearTimers();
    var intro = el('intro');
    if (intro) { intro.pause(); }
    show('intro-clip', false);
    if (stream) {
      stream.getTracks().forEach(function (t) { t.stop(); });
      stream = null;
    }
    var live = el('live');
    if (live) { live.srcObject = null; }
    show('camera-mode', false);
  }

  /** START handler: begin the continuous capture loop on the live feed. */
  function startLoop() {
    if (mode !== 'camera' || !stream) { return; }
    show('cam-controls', false);
    show('cam-instruction', false);
    var live = el('live');
    live.srcObject = stream;
    runCycle();
  }

  /** One loop iteration: 5s live countdown, then grab + analyse a frame. */
  function runCycle() {
    if (mode !== 'camera') { return; }
    /* clear any prior verdict and bring the live preview back */
    var result = el('result');
    if (result) { result.hidden = true; }
    document.body.className = '';
    show('live-wrap', true);
    el('cam-state').textContent = '';

    var remaining = CYCLE_SECONDS;
    var cd = el('countdown');
    cd.textContent = String(remaining);
    cd.hidden = false;
    setStatus('hold a pair of glasses to the camera …', '');

    ticker = window.setInterval(function () {
      remaining -= 1;
      if (remaining > 0) {
        cd.textContent = String(remaining);
        return;
      }
      window.clearInterval(ticker);
      ticker = null;
      cd.hidden = true;
      captureFrame();
    }, 1000);
  }

  /** Grab the current live frame to a canvas and POST it for analysis. */
  function captureFrame() {
    if (mode !== 'camera') { return; }
    var live = el('live');
    var canvas = el('grab');
    var w = live.videoWidth || 1280;
    var h = live.videoHeight || 720;
    canvas.width = w;
    canvas.height = h;
    canvas.getContext('2d').drawImage(live, 0, 0, w, h);
    el('cam-state').textContent = 'captured · analysing …';

    var onDone = function () {
      if (mode !== 'camera') { return; }
      /* hold the verdict briefly, then loop into the next capture */
      later(runCycle, RESULT_HOLD_MS);
    };

    if (canvas.toBlob) {
      canvas.toBlob(function (blob) {
        if (mode !== 'camera') { return; }
        scan(blob, onDone);
      }, 'image/png');
    } else {
      onDone();  // ancient browser without toBlob — just keep the loop alive
    }
  }

  /** Switch input mode and (de)activate the camera accordingly. */
  function setMode(next) {
    if (next === mode) { return; }
    var camTab = el('tab-camera');
    var upTab = el('tab-upload');
    if (next === 'upload') {
      leaveCamera();             // leaveCamera reads `mode`, so flip it after
      mode = 'upload';
      show('upload-mode', true);
      if (camTab) { camTab.className = 'mode-tab'; camTab.setAttribute('aria-selected', 'false'); }
      if (upTab) { upTab.className = 'mode-tab is-active'; upTab.setAttribute('aria-selected', 'true'); }
      var result = el('result');
      if (result) { result.hidden = true; }
      document.body.className = '';
      setStatus('', '');
    } else {
      if (upTab) { upTab.className = 'mode-tab'; upTab.setAttribute('aria-selected', 'false'); }
      if (camTab) { camTab.className = 'mode-tab is-active'; camTab.setAttribute('aria-selected', 'true'); }
      enterCamera();
    }
  }

  /**
   * Wire up the drop zone, file input and reset button.
   */
  function init() {
    var drop = el('drop');
    var input = el('file');
    var again = el('again');

    /* load the classifier weights and the OpenCV.js runtime up front */
    setStatus('loading detector …', 'busy');
    fetch('static/camera_clf.json')
      .then(function (r) { return r.json(); })
      .then(function (m) { model = m; if (cvReady) { setStatus('', ''); } })
      .catch(function () { setStatus('could not load model weights', 'error'); });
    whenCvReady(function () {
      cvReady = true;
      if (model !== null) { setStatus('', ''); }
    });

    if (input) {
      input.addEventListener('change', function () {
        if (input.files && input.files[0]) { scan(input.files[0]); }
      });
    }

    if (drop) {
      drop.addEventListener('keydown', function (ev) {
        if ((ev.key === 'Enter' || ev.key === ' ') && input) { input.click(); }
      });
      drop.addEventListener('dragover', function (ev) {
        ev.preventDefault();
        drop.className = 'drop is-drag';
      });
      drop.addEventListener('dragleave', function () {
        drop.className = 'drop';
      });
      drop.addEventListener('drop', function (ev) {
        ev.preventDefault();
        drop.className = 'drop';
        if (ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files[0]) {
          scan(ev.dataTransfer.files[0]);
        }
      });
    }

    if (again) {
      again.addEventListener('click', function () {
        var result = el('result');
        if (result) { result.hidden = true; }
        document.body.className = '';
        if (input) { input.value = ''; }
        setStatus('', '');
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });
    }

    /* mode tabs */
    var tabCam = el('tab-camera');
    var tabUp = el('tab-upload');
    if (tabCam) { tabCam.addEventListener('click', function () { setMode('camera'); }); }
    if (tabUp) { tabUp.addEventListener('click', function () { setMode('upload'); }); }

    /* START TEST button kicks off the live capture loop */
    var camStart = el('cam-start');
    if (camStart) { camStart.addEventListener('click', startLoop); }

    /* REPLAY restarts the how-it-works clip in the intro column */
    var camRewatch = el('cam-rewatch');
    if (camRewatch) { camRewatch.addEventListener('click', playIntro); }

    /* camera is the default mode on load */
    enterCamera();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
