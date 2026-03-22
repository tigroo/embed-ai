/**
 * YOLO v26 segmentation — ONNX Runtime Web.
 *
 * The ONNX models are exported WITHOUT the end2end NMS head so that
 * all operators are compatible with the WebGL backend.
 *
 * Segmentation output:
 *   output0: [1, 116, 8400] — 4 bbox + 80 class scores + 32 mask coefficients
 *   output1: [1, 32, 160, 160] — 32 mask prototypes at 160×160
 *
 * NMS is performed in JavaScript.  Segmentation masks are computed by
 * multiplying mask coefficients with prototypes, then sigmoid + crop.
 *
 * Backend negotiation (fastest → safest):
 *   1. WebGPU  (if available)
 *   2. WebGL   (mature, broad support)
 *   3. WASM    (universal fallback)
 *
 * Tensor lifecycle: every ort.Tensor wraps WASM/GPU memory outside the
 * JS GC.  We must call .dispose() on every tensor after use.
 */

const INPUT_SIZE = 640;
const NUM_CLASSES = 80;
const NUM_MASK_COEFFS = 32;
const PROTO_H = 160;
const PROTO_W = 160;

let session = null;
let currentModel = null;
let activeBackend = "unknown";

/** True on iOS / iPadOS (all browsers there are WebKit). */
export const isIOS = /iPhone|iPad|iPod/i.test(navigator.userAgent)
  || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);

// Pre-allocated buffer reused every frame.
const _inputBuf = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);

const MODEL_META = {
  fp32: { path: "models/fp32.onnx", size: "~11 MB", label: "FP32" },
  quant: { path: "models/quant.onnx", size: "~4 MB",  label: "Quantized" },
};

/**
 * Ordered backends — fastest first, automatic fallback:
 *   1. WebGPU  (newest, best perf when supported)
 *   2. WebGL   (mature, broad support)
 *   3. WASM    (universal fallback)
 */
function candidateProviders() {
  const list = [];
  if (typeof navigator !== "undefined" && navigator.gpu) list.push("webgpu");
  try {
    const c = document.createElement("canvas");
    if (c.getContext("webgl2") || c.getContext("webgl")) list.push("webgl");
  } catch { /* no WebGL */ }
  list.push("wasm");
  return list;
}

function disposeTensor(t) {
  try { if (t && typeof t.dispose === "function") t.dispose(); } catch { /* ok */ }
}

/**
 * Load (or switch) the ONNX model.
 * @param {"fp32"|"quant"} variant
 */
export async function loadModel(variant) {

  const meta = MODEL_META[variant] ?? MODEL_META.fp32;

  console.log(`[detector] loadModel called with variant='${variant}'`);
  logToServer(`[detector] loadModel called with variant='${variant}'`);

  if (currentModel === variant && session) {
    const logDiv = document.getElementById("backend-log");
    if (logDiv) logDiv.textContent = `Backend: ${activeBackend.toUpperCase()} (${meta.label})`;
    return { backend: activeBackend, size: meta.size, label: meta.label };
  }

  if (session) {
    try { await session.release(); } catch { /* ignore */ }
    session = null;
  }

  let candidates;
  if (variant === "quant") {
    candidates = ["wasm"];
    console.log("[detector] INT8: forcing WASM backend only");
    logToServer("[detector] INT8: forcing WASM backend only");
  } else {
    candidates = candidateProviders();
  }

  console.log(`[detector] Candidates for variant='${variant}': [${candidates}]`);
  logToServer(`[detector] Candidates for variant='${variant}': [${candidates}]`);

  // ...existing code...
  if (variant === "quant" && (candidates.length !== 1 || candidates[0] !== "wasm")) {
    const msg = `[detector] ERROR: INT8 candidates not strictly WASM: [${candidates}]`;
    console.error(msg);
    logToServer(msg);
  }

  console.log(`[detector] Loading ${meta.label} (${meta.path}), candidates: [${candidates}]`);
  logToServer(`[detector] Loading ${meta.label} (${meta.path}), candidates: [${candidates}]`);

  for (const provider of candidates) {
    try {
      console.log(`[detector] trying ${provider} ...`);
      logToServer(`[detector] trying ${provider} ...`);
      const s = await ort.InferenceSession.create(meta.path, {
        executionProviders: [provider],
        graphOptimizationLevel: "all",
      });

      // Preflight — validate the backend can execute the full graph.
      const dummy = new ort.Tensor("float32",
        new Float32Array(3 * INPUT_SIZE * INPUT_SIZE),
        [1, 3, INPUT_SIZE, INPUT_SIZE]);
      let out = null;
      try { out = await s.run({ images: dummy }); }
      finally {
        disposeTensor(dummy);
        if (out) for (const k of Object.keys(out)) disposeTensor(out[k]);
      }

      session = s;
      currentModel = variant;
      activeBackend = provider;
      if (variant === "quant" && provider !== "wasm") {
        const msg = `[detector] ERROR: INT8 model loaded on non-WASM backend: ${provider}`;
        console.error(msg);
        logToServer(msg);
        alert("INT8 model ne doit être utilisé que sur WASM. Rechargement forcé.");
        await session.release();
        session = null;
        continue;
      }
      const logDiv = document.getElementById("backend-log");
      if (logDiv) logDiv.textContent = `Backend: ${provider.toUpperCase()} (${meta.label})`;
      console.log(`[detector] OK ${meta.label} ready on ${provider}`);
      logToServer(`[detector] OK ${meta.label} ready on ${provider}`);
      console.log(`[detector] CONFIRM: activeBackend = ${activeBackend}`);
      logToServer(`[detector] CONFIRM: activeBackend = ${activeBackend}`);
      return { backend: activeBackend, size: meta.size, label: meta.label };
    } catch (err) {
      const logDiv = document.getElementById("backend-log");
      if (logDiv) logDiv.textContent = `Backend FAIL: ${provider} (${err.message ?? err})`;
      console.warn(`[detector] FAIL ${provider}:`, err.message ?? err);
      logToServer(`[detector] FAIL ${provider}: ${err.message ?? err}`);
    }
  }

  throw new Error(`No backend could load ${meta.label}`);
}

/**
 * Run segmentation on a video frame.
 * @param {HTMLVideoElement} video
 * @param {number} confThreshold
 * @returns {{ dets: Array, protos: Float32Array|null, inferMs: number }}
 */
export async function detect(video, confThreshold = 0.35) {
  if (!session) throw new Error("Model not loaded");

  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return { dets: [], protos: null, inferMs: 0 };

  // Prepare input tensor (NCHW, [1,3,640,640], 0-1 float)
  const canvas = getOffscreenCanvas();
  const ctx2d = canvas.getContext("2d", { willReadFrequently: true });
  ctx2d.drawImage(video, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const rgba = ctx2d.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;

  const f = _inputBuf;
  const plane = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0; i < plane; i++) {
    const j = i * 4;
    f[i]             = rgba[j]     / 255;
    f[i + plane]     = rgba[j + 1] / 255;
    f[i + 2 * plane] = rgba[j + 2] / 255;
  }

  const tensor = new ort.Tensor("float32", f, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  let results = null;
  let inferMs = 0;
  try {
    const t0 = performance.now();
    results = await session.run({ images: tensor });
    inferMs = performance.now() - t0;
  } finally {
    disposeTensor(tensor);
  }

  const keys = Object.keys(results);
  let detTensor = null;
  let protoTensor = null;
  for (const k of keys) {
    const t = results[k];
    if (t.dims.length === 4) protoTensor = t;
    else if (t.dims.length === 3) detTensor = t;
  }
  if (!detTensor) detTensor = results[keys[0]];

  const detRaw = detTensor.data;
  const numAnchors = detTensor.dims[2];
  const numRows = detTensor.dims[1];            // 116 for seg, 84 for detect
  const hasMasks = protoTensor && numRows > 84;

  const sx = vw / INPUT_SIZE;
  const sy = vh / INPUT_SIZE;

  const dets = decodeAndNMS(detRaw, numAnchors, numRows, confThreshold, sx, sy, hasMasks);

  // ...existing code...
  let protos = null;
  if (hasMasks && protoTensor) {
    protos = new Float32Array(protoTensor.data);
  }

  for (const k of keys) disposeTensor(results[k]);
  return { dets, protos, inferMs };
}

const NMS_IOU_THRESHOLD = 0.45;
const MAX_DETECTIONS = 300;

/**
 * Decode raw tensor and apply per-class greedy NMS.
 *
 *   rows 0-3  : cx, cy, w, h  (640×640 pixel space)
 *   rows 4-83 : class scores   (post-sigmoid)
 *   rows 84-115: mask coefficients (seg models only)
 */
function decodeAndNMS(raw, numAnchors, numRows, confThreshold, sx, sy, hasMasks) {
  const candidates = [];

  for (let a = 0; a < numAnchors; a++) {
    let bestCls = 0;
    let bestScore = -Infinity;
    for (let c = 0; c < NUM_CLASSES; c++) {
      const score = raw[(4 + c) * numAnchors + a];
      if (score > bestScore) { bestScore = score; bestCls = c; }
    }
    if (bestScore < confThreshold) continue;

    const cx = raw[0 * numAnchors + a];
    const cy = raw[1 * numAnchors + a];
    const w  = raw[2 * numAnchors + a];
    const h  = raw[3 * numAnchors + a];

    const det = {
      x1: (cx - w / 2) * sx,
      y1: (cy - h / 2) * sy,
      x2: (cx + w / 2) * sx,
      y2: (cy + h / 2) * sy,
      conf: bestScore,
      cls: bestCls,
    };

    if (hasMasks) {
      const mc = new Float32Array(NUM_MASK_COEFFS);
      for (let k = 0; k < NUM_MASK_COEFFS; k++) {
        mc[k] = raw[(84 + k) * numAnchors + a];
      }
      det.maskCoeffs = mc;
      det.cx = cx; det.cy = cy; det.bw = w; det.bh = h;
    }

    candidates.push(det);
  }

  candidates.sort((a, b) => b.conf - a.conf);

  const kept = [];
  const suppressed = new Uint8Array(candidates.length);

  for (let i = 0; i < candidates.length && kept.length < MAX_DETECTIONS; i++) {
    if (suppressed[i]) continue;
    const a = candidates[i];
    kept.push(a);

    for (let j = i + 1; j < candidates.length; j++) {
      if (suppressed[j]) continue;
      const b = candidates[j];
      if (b.cls !== a.cls) continue;
      if (iou(a, b) > NMS_IOU_THRESHOLD) suppressed[j] = 1;
    }
  }

  return kept;
}

/**
 * Compute a binary mask for one detection.
 *
 * @param {Object} det  — detection with .maskCoeffs and .cx/.cy/.bw/.bh
 * @param {Float32Array} protos — [32 × 160 × 160] flat array
 * @returns {Uint8Array} — PROTO_H × PROTO_W mask (0 or 255), cropped to bbox
 */
export function computeMask(det, protos) {
  const mask = new Uint8Array(PROTO_H * PROTO_W);
  const mc = det.maskCoeffs;
  if (!mc || !protos) return mask;

  const protoScale = PROTO_H / INPUT_SIZE;
  const bx1 = Math.max(0, Math.floor((det.cx - det.bw / 2) * protoScale));
  const by1 = Math.max(0, Math.floor((det.cy - det.bh / 2) * protoScale));
  const bx2 = Math.min(PROTO_W, Math.ceil((det.cx + det.bw / 2) * protoScale));
  const by2 = Math.min(PROTO_H, Math.ceil((det.cy + det.bh / 2) * protoScale));

  const ppx = PROTO_H * PROTO_W;   // pixels per prototype plane
  for (let y = by1; y < by2; y++) {
    for (let x = bx1; x < bx2; x++) {
      let val = 0;
      const idx = y * PROTO_W + x;
      for (let k = 0; k < NUM_MASK_COEFFS; k++) {
        val += mc[k] * protos[k * ppx + idx];
      }
      // sigmoid
      mask[idx] = (1 / (1 + Math.exp(-val))) > 0.5 ? 255 : 0;
    }
  }

  return mask;
}

export { PROTO_H, PROTO_W, INPUT_SIZE as MODEL_INPUT_SIZE };

function iou(a, b) {
  const ix1 = Math.max(a.x1, b.x1);
  const iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2);
  const iy2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-6);
}

let _osc = null;
function getOffscreenCanvas() {
  if (!_osc) { _osc = document.createElement("canvas"); _osc.width = _osc.height = INPUT_SIZE; }
  return _osc;
}

function logToServer(msg) {
  try {
    fetch("/log_js", {
      method: "POST",
      headers: { "Content-Type": "text/plain" },
      body: msg,
    });
  } catch (e) {
    // ignore
  }
}
