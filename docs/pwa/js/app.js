/**
 * Main application — camera, render loop, HUD, UI controls.
 */
import { loadModel, detect, computeMask, isIOS, PROTO_H, PROTO_W, MODEL_INPUT_SIZE } from "./detector.js";
import { COCO_LABELS, classColor } from "./coco-labels.js";

// ── Global crash guards ─────────────────────────────────────────────────────
// Catch anything that would otherwise kill the page on iOS WebKit.
window.addEventListener("error", (e) => {
  console.error("[global]", e.message);
  e.preventDefault();          // stop WebKit from tearing down the page
});
window.addEventListener("unhandledrejection", (e) => {
  console.error("[promise]", e.reason);
  e.preventDefault();
});

// ── DOM refs ────────────────────────────────────────────────────────────────
const video    = document.getElementById("camera");
const overlay  = document.getElementById("overlay");
const ctx      = overlay.getContext("2d");
const statusEl = document.getElementById("status");
const modelSel = document.getElementById("model-select");
const confIn   = document.getElementById("conf-slider");
const confVal  = document.getElementById("conf-value");

const hFps   = document.getElementById("h-fps");
const hInf   = document.getElementById("h-inf");
const hObj   = document.getElementById("h-obj");
const hModel = document.getElementById("h-model");
const hSize  = document.getElementById("h-size");
const hBack  = document.getElementById("h-back");

let running = false;
let confThreshold = 0.35;
let loopBusy = false;
let consecutiveErrors = 0;
const MAX_CONSECUTIVE_ERR = 5;

let smoothFps = 0;
const FPS_ALPHA = 0.15;
const MIN_FRAME_GAP_MS = 30;

// Track whether the model has been loaded at least once (for resume).
let modelLoaded = false;

// Anti-bounce: minimum interval between camera restarts (ms).
// On iOS WebKit, too-frequent getUserMedia calls trigger a permission
// re-prompt loop that effectively kills the page.
const RESUME_COOLDOWN_MS = isIOS ? 3000 : 1000;
let lastResumeTs = 0;
let resumeTimer = null;

// ── Camera ──────────────────────────────────────────────────────────────────

async function startCamera() {
  statusEl.textContent = "Starting camera\u2026";
  try {
    stopCamera(); // release any previous stream first

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    overlay.width  = video.videoWidth;
    overlay.height = video.videoHeight;
    statusEl.textContent = "";
    return true;
  } catch (err) {
    statusEl.textContent = `Camera: ${err.message}`;
    console.error("Camera error:", err);
    return false;
  }
}

/** Release the camera hardware immediately — critical before iOS hides page. */
function stopCamera() {
  try {
    if (video.srcObject) {
      video.srcObject.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    }
  } catch { /* best effort */ }
}

/** Is the camera stream alive? */
function cameraAlive() {
  try {
    const tracks = video.srcObject && video.srcObject.getVideoTracks();
    return tracks && tracks.length > 0 && tracks[0].readyState === "live";
  } catch { return false; }
}

// ── Render loop ─────────────────────────────────────────────────────────────
function scheduleNext() {
  if (!running) return;
  setTimeout(() => requestAnimationFrame(loop), MIN_FRAME_GAP_MS);
}

async function loop() {
  if (!running || loopBusy) return;
  loopBusy = true;
  const t0 = performance.now();

  try {
    if (!cameraAlive()) {
      // Camera died mid-loop — stop cleanly, schedule a debounced restart.
      loopBusy = false;
      running = false;
      statusEl.textContent = "Camera lost — resuming\u2026";
      scheduleResume();
      return;
    }

    const { dets, protos, inferMs } = await detect(video, confThreshold);
    drawDetections(dets, protos);

    const totalMs = performance.now() - t0;
    const fps = 1000 / totalMs;
    smoothFps = smoothFps === 0 ? fps : smoothFps * (1 - FPS_ALPHA) + fps * FPS_ALPHA;

    hFps.textContent = smoothFps.toFixed(1);
    hInf.textContent = inferMs.toFixed(0) + " ms";
    hObj.textContent = dets.length;
    consecutiveErrors = 0;
  } catch (err) {
    console.error("Loop:", err);
    if (++consecutiveErrors >= MAX_CONSECUTIVE_ERR) {
      running = false;
      statusEl.textContent = `Stopped — ${err.message}`;
      loopBusy = false;
      return;
    }
  }

  loopBusy = false;
  scheduleNext();
}

// ── Drawing (segmentation masks + bounding boxes) ───────────────────────────

// Offscreen canvas for mask compositing at prototype resolution.
let _maskCanvas = null;
let _maskCtx = null;
function getMaskCanvas() {
  if (!_maskCanvas) {
    _maskCanvas = document.createElement("canvas");
    _maskCanvas.width = PROTO_W;
    _maskCanvas.height = PROTO_H;
    _maskCtx = _maskCanvas.getContext("2d", { willReadFrequently: true });
  }
  return { c: _maskCanvas, cx: _maskCtx };
}

const MASK_ALPHA = 0.40;

function drawDetections(dets, protos) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const scale = Math.max(1, overlay.width / 640);
  const hasMasks = protos && dets.length > 0 && dets[0].maskCoeffs;

  // ── 1. Draw segmentation masks ─────────────────────────────────────
  if (hasMasks) {
    const { c: mc, cx: mctx } = getMaskCanvas();
    // Scale from proto space (160×160) to overlay (video) space
    const sxm = overlay.width / MODEL_INPUT_SIZE;
    const sym = overlay.height / MODEL_INPUT_SIZE;

    ctx.save();
    ctx.globalAlpha = MASK_ALPHA;

    for (const d of dets) {
      const mask = computeMask(d, protos);

      // Build an ImageData at proto resolution for this mask
      const imgData = mctx.createImageData(PROTO_W, PROTO_H);
      const rgba = imgData.data;
      const color = classColor(d.cls);
      // Parse hex color
      const r = parseInt(color.slice(1, 3), 16);
      const g = parseInt(color.slice(3, 5), 16);
      const b = parseInt(color.slice(5, 7), 16);

      for (let i = 0; i < mask.length; i++) {
        if (mask[i]) {
          const off = i * 4;
          rgba[off]     = r;
          rgba[off + 1] = g;
          rgba[off + 2] = b;
          rgba[off + 3] = 255;
        }
      }
      mctx.putImageData(imgData, 0, 0);

      // Crop region in proto space → overlay space
      const protoScale = PROTO_H / MODEL_INPUT_SIZE;
      const px1 = Math.max(0, (d.cx - d.bw / 2) * protoScale) | 0;
      const py1 = Math.max(0, (d.cy - d.bh / 2) * protoScale) | 0;
      const px2 = Math.min(PROTO_W, (d.cx + d.bw / 2) * protoScale + 1) | 0;
      const py2 = Math.min(PROTO_H, (d.cy + d.bh / 2) * protoScale + 1) | 0;
      const pw = px2 - px1;
      const ph = py2 - py1;
      if (pw > 0 && ph > 0) {
        ctx.drawImage(mc,
          px1, py1, pw, ph,                                    // src crop
          px1 / protoScale * sxm, py1 / protoScale * sym,     // dst pos
          pw / protoScale * sxm, ph / protoScale * sym,        // dst size
        );
      }
    }

    ctx.restore();
  }

  // ── 2. Draw bounding boxes + labels ────────────────────────────────
  for (const d of dets) {
    const color = classColor(d.cls);
    const label = `${COCO_LABELS[d.cls] ?? d.cls} ${(d.conf * 100).toFixed(0)}%`;
    const bw = d.x2 - d.x1;
    const bh = d.y2 - d.y1;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2 * scale;
    ctx.strokeRect(d.x1, d.y1, bw, bh);
    const fontSize = Math.round(13 * scale);
    ctx.font = `bold ${fontSize}px sans-serif`;
    const tw = ctx.measureText(label).width + 8 * scale;
    const lh = fontSize + 6 * scale;
    ctx.fillStyle = color;
    ctx.fillRect(d.x1, d.y1 - lh, tw, lh);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, d.x1 + 4 * scale, d.y1 - 4 * scale);
  }
}

function updateHudModel(info) {
  hModel.textContent = info.label;
  hSize.textContent  = info.size;
  hBack.textContent  = info.backend;
  try {
    fetch("/log_backend", {
      method: "POST",
      headers: { "Content-Type": "text/plain" },
      body: `${info.label} | ${info.backend}`
    });
  } catch {}
}

// ── Pause / Resume (iOS lifecycle) ──────────────────────────────────────────

function pauseAll() {
  console.log("⏸ pause — releasing camera");
  running = false;
  loopBusy = false;
  if (resumeTimer) { clearTimeout(resumeTimer); resumeTimer = null; }
  stopCamera();                   // FREE the hardware so iOS doesn't kill us
}

/**
 * Schedule a debounced resume.  Prevents rapid getUserMedia() calls
 * that trigger the iOS "allow camera?" prompt loop.
 */
function scheduleResume() {
  if (resumeTimer) return;        // already scheduled
  const elapsed = performance.now() - lastResumeTs;
  const wait = Math.max(0, RESUME_COOLDOWN_MS - elapsed);
  console.log(`⏳ scheduling resume in ${Math.round(wait)} ms`);
  resumeTimer = setTimeout(() => {
    resumeTimer = null;
    resumeAll();
  }, wait);
}

async function resumeAll() {
  console.log("▶ resume — restarting camera + loop");
  lastResumeTs = performance.now();
  smoothFps = 0;
  consecutiveErrors = 0;

  const ok = await startCamera();
  if (!ok) return;                // camera refused (user denied?) — stay paused

  // Model is still in WASM memory — no need to reload it.
  if (modelLoaded) {
    running = true;
    scheduleNext();
  }
}

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    pauseAll();
  } else {
    scheduleResume();
  }
});

// bfcache restore (Safari back/forward cache)
window.addEventListener("pageshow", (e) => {
  if (e.persisted) {
    console.log("pageshow (bfcache restore)");
    scheduleResume();
  }
});

// ── UI events ───────────────────────────────────────────────────────────────

modelSel.addEventListener("change", async () => {
  running = false;
  loopBusy = false;
  smoothFps = 0;
  consecutiveErrors = 0;
  statusEl.textContent = "Loading model\u2026";
  try {
    if (!cameraAlive()) await startCamera();
    const info = await loadModel(modelSel.value);
    updateHudModel(info);
    modelLoaded = true;
    statusEl.textContent = "";
    running = true;
    scheduleNext();
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
    console.error("Model switch:", err);
  }
});

confIn.addEventListener("input", () => {
  confThreshold = parseFloat(confIn.value);
  confVal.textContent = confThreshold.toFixed(2);
});

// ── Boot ────────────────────────────────────────────────────────────────────

(async function init() {
  try {
    await startCamera();
    statusEl.textContent = "Loading model\u2026";
    const info = await loadModel(modelSel.value);
    updateHudModel(info);
    modelLoaded = true;
    statusEl.textContent = "";
    running = true;
    scheduleNext();
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
    console.error(err);
  }
})();

// ── PWA install prompt ──────────────────────────────────────────────────────

let deferredPrompt = null;
window.addEventListener("beforeinstallprompt", (e) => {
  e.preventDefault();
  deferredPrompt = e;
  const btn = document.getElementById("install-btn");
  if (btn) {
    btn.style.display = "block";
    btn.addEventListener("click", async () => {
      deferredPrompt.prompt();
      await deferredPrompt.userChoice;
      deferredPrompt = null;
      btn.style.display = "none";
    }, { once: true });
  }
});

// ── Service Worker ──────────────────────────────────────────────────────────
if ("serviceWorker" in navigator) {
  if (isIOS) {
    navigator.serviceWorker.getRegistrations().then((regs) => {
      regs.forEach((r) => {
        r.unregister();
        console.log("Unregistered SW:", r.scope);
      });
    });
    if ("caches" in window) {
      caches.keys().then((keys) => keys.forEach((k) => caches.delete(k)));
    }
  } else {
    navigator.serviceWorker.register("sw.js").catch(console.error);
  }
}
