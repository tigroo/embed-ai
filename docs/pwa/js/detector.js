/**
 * YOLO v26 detector — ONNX Runtime Web.
 *
 * Backend negotiation (fastest → safest):
 *   1. WebGPU  (if available)
 *   2. WebGL
 *   3. WASM
 *
 * Tensor lifecycle: every ort.Tensor wraps WASM/GPU memory outside the
 * JS GC.  We must call .dispose() on every tensor after use.
 */

const INPUT_SIZE = 640;

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
 * Ordered backends — WebGL first (mature & stable on all platforms
 * including iOS), then WebGPU (newer, still buggy on iOS WebKit),
 * then WASM as ultimate fallback.
 */
function candidateProviders() {
  const list = [];
  try {
    const c = document.createElement("canvas");
    if (c.getContext("webgl2") || c.getContext("webgl")) list.push("webgl");
  } catch { /* no WebGL */ }
  if (typeof navigator !== "undefined" && navigator.gpu) list.push("webgpu");
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
  console.log(`Loading ${meta.label} (${meta.path}), trying: [${candidates}] …`);

  for (const provider of candidates) {
    try {
      const s = await ort.InferenceSession.create(meta.path, {
        executionProviders: [provider],
        graphOptimizationLevel: "all",
      });

      // Preflight — validate that the backend can actually execute the graph
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
      console.log(`✔ ${meta.label} ready on ${provider}`);
      return { backend: activeBackend, size: meta.size, label: meta.label };
    } catch (err) {
      console.warn(`✘ ${provider} failed: ${err.message}`);
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

  const output = results[Object.keys(results)[0]];
  const raw = output.data;
  const numDet = output.dims[1];
  const sx = vw / INPUT_SIZE;
  const sy = vh / INPUT_SIZE;

  const dets = [];
  for (let i = 0; i < numDet; i++) {
    const off = i * 6;
    const conf = raw[off + 4];
    if (conf < confThreshold) continue;
    dets.push({
      x1: raw[off] * sx, y1: raw[off + 1] * sy,
      x2: raw[off + 2] * sx, y2: raw[off + 3] * sy,
      conf, cls: Math.round(raw[off + 5]),
    });
  }

  for (const k of Object.keys(results)) disposeTensor(results[k]);
  return { dets, inferMs };
}

let _osc = null;
function getOffscreenCanvas() {
  if (!_osc) { _osc = document.createElement("canvas"); _osc.width = _osc.height = INPUT_SIZE; }
  return _osc;
}
