/* Node parity harness: run the JS OpenCV pipeline over the SAME cv2-decoded
 * pixels Python used, and compare crops + verdicts. A mismatch here is a port
 * bug, because the decoder is identical by construction (tools/dump_raw.py feeds
 * cv2-decoded RGBA). Browser-decoder robustness is checked separately via
 * browser_test.html in a real browser.
 *
 * Prereqs:  python tools/dump_crops.py   &&   python tools/dump_raw.py
 * Run:      node tools/parity_node.mjs
 */
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const require = createRequire(import.meta.url);
const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');

// Load OpenCV.js FIRST (before the detector modules) and wait for the wasm
// runtime; requiring it after other modules can leave onRuntimeInitialized unset.
let cv = require(join(ROOT, 'static/opencv.js'));
if (cv instanceof Promise) { cv = await cv; }
if (!cv.Mat) {
  await new Promise((resolve) => {
    cv.onRuntimeInitialized = resolve;
    setTimeout(resolve, 20000);
  });
}
if (!cv.Mat) { console.error('OpenCV.js failed to initialise'); process.exit(1); }

const D = (n) => join(ROOT, 'static/detector', n);
const config = require(D('config.js'));
const preprocess = require(D('preprocess.js')).preprocess;
const segment = require(D('segment.js')).segment;
const locate = require(D('locate.js')).locate;
const clf = require(D('camera_clf.js'));
const decide = require(D('decide.js')).decide;

const model = JSON.parse(readFileSync(join(ROOT, 'static/camera_clf.json')));
const reference = JSON.parse(readFileSync(join(ROOT, 'tools/parity/reference.json')));
const manifest = JSON.parse(readFileSync(join(ROOT, 'tools/parity/raw/manifest.json')));

const meanAbsDiff = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) s += Math.abs(a[i] - b[i]);
  return s / a.length;
};

let verdictAgree = 0;
let cropOk = 0;
const cropDiffs = [];
const flips = [];

for (const m of manifest) {
  const ref = reference[m.i];
  const buf = readFileSync(join(ROOT, `tools/parity/raw/${m.i}.bin`));
  const data = new Uint8ClampedArray(buf.buffer, buf.byteOffset, buf.length);
  const src = cv.matFromImageData({ data, width: m.w, height: m.h });

  const pre = preprocess(src, config, cv);
  const seg = segment(pre.grayEq, config, cv);
  const loc = locate(pre.grayEq, seg, config, cv);

  const per = {};
  let worstCrop = 0;
  for (const roi of loc.rois) {
    const crop = clf.cornerCrop(pre.gray, roi, cv);
    per[roi.side] = Math.round(clf.probFromCrop(crop, model) * 1000) / 1000;
    const refCorner = ref.corners.find((c) => c.side === roi.side);
    if (refCorner) {
      const d = meanAbsDiff(crop, refCorner.crop);
      if (d > worstCrop) worstCrop = d;
    }
  }
  const v = decide(per.L, per.R, seg.isolated(), config);
  if (v.verdict === ref.verdict) verdictAgree += 1;
  else flips.push(`${ref.name}: py=${ref.verdict} js=${v.verdict} [L=${per.L} R=${per.R}]`);

  cropDiffs.push(worstCrop);
  if (worstCrop <= 1.0) cropOk += 1;

  seg.free(); pre.free(); src.delete();
}

const N = manifest.length;
const avgDiff = cropDiffs.reduce((a, b) => a + b, 0) / N;
const maxDiff = Math.max(...cropDiffs);
console.log(`Crop parity (mean abs pixel diff vs Python): avg=${avgDiff.toFixed(3)}, max=${maxDiff.toFixed(3)}`);
console.log(`Crops within 1.0/px of Python: ${cropOk}/${N}`);
console.log(`Verdict agreement vs Python (full JS pipeline): ${verdictAgree}/${N}`);
flips.forEach((s) => console.log('  FLIP ' + s));
