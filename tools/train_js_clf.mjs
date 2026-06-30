/* Retrain the corner-camera logistic regression in the JS-HOG feature space.
 *
 * The browser uses static/detector/hog.js (not cv2's HOG), so the shipped model
 * must be trained on that descriptor. This is a faithful port of
 * pipeline/camera_clf.train_logreg (full-batch GD, L2, no bias reg) over the
 * augmented crops Python emitted in tools/parity/train_crops.json.
 *
 * Outputs static/camera_clf.json (overwriting the cv2-space weights) and prints
 * leave-one-image-out CV (both-corners rule) plus reference-set verdict agreement
 * against Python, using tools/parity/reference.json.
 *
 *     node tools/train_js_clf.mjs
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const require = createRequire(import.meta.url);
const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const hog = require(join(ROOT, 'static/detector/hog.js'));

const DIM = hog.DIM;
const THRESH = 0.5;

function hogOf(px) { return hog.hogFeat(px); }

/** standardise + sigmoid logit — same math as CameraClf.prob */
function prob(feat, m) {
  let z = m.b;
  for (let i = 0; i < DIM; i += 1) z += ((feat[i] - m.mu[i]) / m.sd[i]) * m.w[i];
  return 1 / (1 + Math.exp(-z));
}

/** Port of camera_clf.train_logreg: standardise X, full-batch GD, L2 (no bias). */
function trainLogreg(X, y, l2 = 1.0, lr = 0.1, iters = 2000) {
  const n = X.length;
  const mu = new Float64Array(DIM);
  const sd = new Float64Array(DIM);
  for (let j = 0; j < DIM; j += 1) {
    let s = 0;
    for (let i = 0; i < n; i += 1) s += X[i][j];
    mu[j] = s / n;
  }
  for (let j = 0; j < DIM; j += 1) {
    let s = 0;
    for (let i = 0; i < n; i += 1) { const d = X[i][j] - mu[j]; s += d * d; }
    sd[j] = Math.sqrt(s / n) + 1e-6;
  }
  const Xn = X.map((row) => {
    const r = new Float64Array(DIM);
    for (let j = 0; j < DIM; j += 1) r[j] = (row[j] - mu[j]) / sd[j];
    return r;
  });
  const w = new Float64Array(DIM);
  let b = 0;
  for (let it = 0; it < iters; it += 1) {
    const gw = new Float64Array(DIM);
    let gb = 0;
    for (let i = 0; i < n; i += 1) {
      let z = b;
      const row = Xn[i];
      for (let j = 0; j < DIM; j += 1) z += row[j] * w[j];
      const p = 1 / (1 + Math.exp(-z));
      const g = p - y[i];
      for (let j = 0; j < DIM; j += 1) gw[j] += g * row[j];
      gb += g;
    }
    for (let j = 0; j < DIM; j += 1) { gw[j] = gw[j] / n + (l2 * w[j]) / n; w[j] -= lr * gw[j]; }
    b -= lr * (gb / n);
  }
  return { w: Array.from(w), b, mu: Array.from(mu), sd: Array.from(sd) };
}

// ---- load data ----
const train = JSON.parse(readFileSync(join(ROOT, 'tools/parity/train_crops.json')));
const reference = JSON.parse(readFileSync(join(ROOT, 'tools/parity/reference.json')));

const feats = train.map((t) => hogOf(t.pixels));
const labels = train.map((t) => t.label);
const groups = train.map((t) => t.group);

// ---- leave-one-image-out CV (both-corners rule, mirrors train_camera_clf) ----
let correct = 0;
const misses = [];
for (let gi = 0; gi < reference.length; gi += 1) {
  const X = [], y = [];
  for (let i = 0; i < feats.length; i += 1) {
    if (groups[i] === gi) continue;
    X.push(feats[i]); y.push(labels[i]);
  }
  const m = trainLogreg(X, y);
  const ref = reference[gi];
  const probs = ref.corners.map((c) => prob(hogOf(c.crop), m));
  const pred = probs.every((p) => p >= THRESH) ? 1 : 0;
  if (pred === ref.label) correct += 1;
  else misses.push(`${ref.label === 1 ? 'camera->missed' : 'normal->FALSEPOS'} ${ref.name} [${probs.map((p) => p.toFixed(2)).join(', ')}]`);
}
console.log(`LEAVE-ONE-IMAGE-OUT CV (both-corners): ${correct}/${reference.length} = ${(100 * correct / reference.length).toFixed(0)}%`);
misses.forEach((s) => console.log('  MISS ' + s));

// ---- final model on ALL crops ----
const model = trainLogreg(feats, labels);
model.thresh = THRESH;
writeFileSync(join(ROOT, 'static/camera_clf.json'), JSON.stringify(model));
console.log(`\nwrote static/camera_clf.json (dim=${DIM}, b=${model.b.toFixed(4)})`);

// ---- reference-set verdict agreement vs Python (in-sample) ----
let agree = 0;
const flips = [];
for (const ref of reference) {
  const probs = ref.corners.map((c) => prob(hogOf(c.crop), model));
  const jsVerdict = probs.every((p) => p >= THRESH) ? 'META' : 'NORMAL';
  if (jsVerdict === ref.verdict) agree += 1;
  else flips.push(`${ref.name}: py=${ref.verdict} js=${jsVerdict} [${probs.map((p) => p.toFixed(2)).join(', ')}] (py [${ref.corners.map((c) => (c.cam_prob == null ? 'na' : c.cam_prob.toFixed(2))).join(', ')}])`);
}
console.log(`\nReference verdict agreement vs Python (same crops): ${agree}/${reference.length}`);
flips.forEach((s) => console.log('  FLIP ' + s));
