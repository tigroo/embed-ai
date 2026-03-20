/**
 * YOLO v26 detector — ONNX Runtime Web.
 *
 * Backend negotiation (fastest → safest):
 *   1. WebGPU  (if available)
 *   2. WebGL
 *   3. WASM
 *
 * We try each backend by actually creating a session; if the backend
 * rejects the model (missing op, GPU init failure…) we fall through.
 *
 * IMPORTANT — tensor lifecycle:
 *   Every ort.Tensor holds a chunk of WASM linear memory that is NOT
 *   reached by the JS garbage collector.  We MUST call .dispose() on
 *   input tensors and output tensors after each inference, otherwise
 *   the WASM heap grows until the OS kills the tab (especially iOS).
 */

const INPUT_SIZE = 640;

let session = null;
let currentModel = null;
let activeBackend = "unknown";

// Pre-allocated buffer reused every frame (avoids ~5 MB alloc per frame).
const _inputBuf = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);

const MODEL_META = {
  fp32: { path: "models/yolo26n_fp32.onnx", size: "~10 MB", label: "FP32" },
  int8: { path: "models/yolo26n_int8.onnx",  size: "~3 MB",  label: "INT8" },
};

/**
 * Ordered list of backends to try.
 * @param {"fp32"|"int8"} variant
 */
function candidateProviders(variant) {
  const list = [];

  if (typeof navigator !== "undefined" && navigator.gpu) {
    list.push("webgpu");
  }

  try {
    const c = document.createElement("canvas");
    if (c.getContext("webgl2") || c.getContext("webgl")) {
      list.push("webgl");
    }
  } catch { /* no WebGL */ }

  list.push("wasm");
  return list;
}

/**
 * Safely dispose an ort.Tensor (no-op if null / already released).
 */
function disposeTensor(t) {
  try { if (t && typeof t.dispose === "function") t.dispose(); } catch { /* ignore */ }
}

/**
 * Load (or switch) the ONNX model.
 * Tries backends in order; first successful session wins.
 *
 * @param {"fp32"|"int8"} variant
 * @returns {Promise<{backend: string, size: string, label: string}>}
 */
export async function loadModel(variant) {
  const meta = MODEL_META[variant] ?? MODEL_META.fp32;

  if (currentModel === variant && session) {
    return { backend: activeBackend, size: meta.size, label: meta.label };
  }

  // Dispose previous session
  if (session) {
    try { await session.release(); } catch { /* ignore */ }
    session = null;
  }

  const candidates = candidateProviders(variant);
  if (!candidates.length) {
    throw new Error(`No compatible backend for ${meta.label} on this device`);
  }
  console.log(`Loading ${meta.label} (${meta.path}), trying: [${candidates}] …`);

  for (const provider of candidates) {
    try {
      const opts = {
        executionProviders: [provider],
        graphOptimizationLevel: "all",
      };
      const s = await ort.InferenceSession.create(meta.path, opts);

      // Preflight: run one dummy inference to catch backends that load
      // the graph but crash on unsupported ops during execution.
      const dummy = new ort.Tensor(
        "float32",
        new Float32Array(3 * INPUT_SIZE * INPUT_SIZE),
        [1, 3, INPUT_SIZE, INPUT_SIZE],
      );
      let preflight = null;
      try {
        preflight = await s.run({ images: dummy });
      } finally {
        disposeTensor(dummy);
        if (preflight) {
          for (const k of Object.keys(preflight)) disposeTensor(preflight[k]);
        }
      }

      session = s;
      currentModel = variant;
      activeBackend = provider;
      console.log(`✔ ${meta.label} loaded + verified on ${provider}`);
      return { backend: activeBackend, size: meta.size, label: meta.label };
    } catch (err) {
      console.warn(`✘ ${provider} failed for ${meta.label}: ${err.message}`);
    }
  }

  throw new Error(`No backend could load ${meta.label}`);
}


/**
 * Run detection on a video element.
 * @param {HTMLVideoElement} video
 * @param {number} confThreshold
 * @returns {Promise<{dets: Array, inferMs: number}>}
 */
export async function detect(video, confThreshold = 0.35) {
  if (!session) throw new Error("Model not loaded");

  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return { dets: [], inferMs: 0 };

  // Pre-process: resize to 640×640 and convert RGBA HWC uint8 → RGB CHW float32 [0,1]
  const canvas = getOffscreenCanvas();
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(video, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);

  const float32 = _inputBuf;          // reuse pre-allocated buffer
  const plane = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0; i < plane; i++) {
    const j = i * 4;
    float32[i]             = data[j]     / 255;  // R
    float32[i + plane]     = data[j + 1] / 255;  // G
    float32[i + 2 * plane] = data[j + 2] / 255;  // B
  }

  const tensor = new ort.Tensor("float32", float32, [1, 3, INPUT_SIZE, INPUT_SIZE]);

  let results = null;
  let inferMs = 0;
  try {
    const t0 = performance.now();
    results = await session.run({ images: tensor });
    inferMs = performance.now() - t0;
  } finally {
    disposeTensor(tensor);
  }

  // Output shape: [1, 300, 6] → [x1, y1, x2, y2, conf, classId]
  const output = results[Object.keys(results)[0]];
  const raw = output.data;
  const numDet = output.dims[1]; // 300

  const sx = vw / INPUT_SIZE;
  const sy = vh / INPUT_SIZE;

  const dets = [];
  for (let i = 0; i < numDet; i++) {
    const off = i * 6;
    const conf = raw[off + 4];
    if (conf < confThreshold) continue;
    dets.push({
      x1:  raw[off]     * sx,
      y1:  raw[off + 1] * sy,
      x2:  raw[off + 2] * sx,
      y2:  raw[off + 3] * sy,
      conf,
      cls: Math.round(raw[off + 5]),
    });
  }

  // Dispose ALL output tensors to free WASM memory
  for (const k of Object.keys(results)) disposeTensor(results[k]);

  return { dets, inferMs };
}

// ── Reusable offscreen canvas ───────────────────────────────────────────────
let _osc = null;
function getOffscreenCanvas() {
  if (!_osc) {
    _osc = document.createElement("canvas");
    _osc.width = INPUT_SIZE;
    _osc.height = INPUT_SIZE;
  }
  return _osc;
}
