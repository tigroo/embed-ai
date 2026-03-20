/**
 * YOLO v26 detector — ONNX Runtime Web.
 *
 * The ONNX models are exported WITHOUT the end2end NMS head so that
 * all operators are compatible with the WebGL backend.  Output shape
 * is [1, 84, 8400] (4 bbox coords + 80 class scores × 8400 anchors).
 * NMS is performed here in JavaScript.
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
const NUM_ANCHORS = 8400;

let session = null;
let currentModel = null;
let activeBackend = "unknown";

/** True on iOS / iPadOS (all browsers there are WebKit). */
export const isIOS = /iPhone|iPad|iPod/i.test(navigator.userAgent)
  || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);

// Pre-allocated buffer reused every frame.
const _inputBuf = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);

const MODEL_META = {
  fp32: { path: "models/yolo26n_fp32.onnx", size: "~10 MB", label: "FP32" },
  int8: { path: "models/yolo26n_int8.onnx",  size: "~3 MB",  label: "INT8" },
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
 * @param {"fp32"|"int8"} variant
 */
export async function loadModel(variant) {
  const meta = MODEL_META[variant] ?? MODEL_META.fp32;

  if (currentModel === variant && session) {
    return { backend: activeBackend, size: meta.size, label: meta.label };
  }

  if (session) {
    try { await session.release(); } catch { /* ignore */ }
    session = null;
  }

  const candidates = candidateProviders();
  console.log(`[detector] Loading ${meta.label} (${meta.path}), candidates: [${candidates}]`);

  for (const provider of candidates) {
    try {
      console.log(`[detector] trying ${provider} ...`);
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
      console.log(`[detector] OK ${meta.label} ready on ${provider}`);
      return { backend: activeBackend, size: meta.size, label: meta.label };
    } catch (err) {
      console.warn(`[detector] FAIL ${provider}:`, err.message ?? err);
    }
  }

  throw new Error(`No backend could load ${meta.label}`);
}

/**
 * Run detection on a video frame.
 * @param {HTMLVideoElement} video
 * @param {number} confThreshold
 */
export async function detect(video, confThreshold = 0.35) {
  if (!session) throw new Error("Model not loaded");

  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return { dets: [], inferMs: 0 };

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

  // Parse raw output [1, 84, 8400] and run NMS
  const output = results[Object.keys(results)[0]];
  const raw = output.data;
  const sx = vw / INPUT_SIZE;
  const sy = vh / INPUT_SIZE;

  const dets = decodeAndNMS(raw, confThreshold, sx, sy);

  for (const k of Object.keys(results)) disposeTensor(results[k]);
  return { dets, inferMs };
}

// ── Raw [84, 8400] → detections + greedy NMS ─────────────────────────────

const NMS_IOU_THRESHOLD = 0.45;
const MAX_DETECTIONS = 300;

/**
 * Decode [84, 8400] raw tensor and apply per-class greedy NMS.
 *
 * Layout (row-major in the flat array):
 *   raw[row * 8400 + anchor]
 *   rows 0-3 : cx, cy, w, h  (640×640 pixel space)
 *   rows 4-83: class scores   (post-sigmoid)
 */
function decodeAndNMS(raw, confThreshold, sx, sy) {
  const candidates = [];

  for (let a = 0; a < NUM_ANCHORS; a++) {
    let bestCls = 0;
    let bestScore = -Infinity;
    for (let c = 0; c < NUM_CLASSES; c++) {
      const score = raw[(4 + c) * NUM_ANCHORS + a];
      if (score > bestScore) { bestScore = score; bestCls = c; }
    }
    if (bestScore < confThreshold) continue;

    const cx = raw[0 * NUM_ANCHORS + a];
    const cy = raw[1 * NUM_ANCHORS + a];
    const w  = raw[2 * NUM_ANCHORS + a];
    const h  = raw[3 * NUM_ANCHORS + a];

    candidates.push({
      x1: (cx - w / 2) * sx,
      y1: (cy - h / 2) * sy,
      x2: (cx + w / 2) * sx,
      y2: (cy + h / 2) * sy,
      conf: bestScore,
      cls: bestCls,
    });
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
