/**
 * Main application — camera, render loop, HUD, UI controls.
 *
 * iOS resilience
 * ──────────────
 * All iOS browsers use WebKit.  WebKit aggressively kills ("jetsams")
 * web-content processes that hold resources while the page is hidden
 * (notification shade, control centre, tab switch, lock screen…).
 * When the process is killed the page reloads → camera re-prompted.
 *
 * Our defence:
 *   • Listen for `visibilitychange` — STOP inference AND release the
 *     camera stream the instant the page becomes hidden.
 *   • On `visibilitychange` visible → restart camera + loop.
 *   • On `pageshow` with `persisted` (bfcache restore) → same.
 *   • Global error/rejection guards so nothing crashes the page.
 *   • No Service Worker registration on iOS (removes a reload vector).
 */
import { loadModel, detect, isIOS } from "./detector.js";
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

// ── Camera ──────────────────────────────────────────────────────────────────

async function startCamera() {
  statusEl.textContent = "Starting camera\u2026";
  try {
    stopCamera();                 // release any previous stream first

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
      // Camera died mid-loop — stop cleanly, the visibilitychange handler
      // or user action will restart it.
      loopBusy = false;
      running = false;
      statusEl.textContent = "Camera lost — resuming\u2026";
      resumeAll();
      return;
    }

    const { dets, inferMs } = await detect(video, confThreshold);
    drawDetections(dets);

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

// ── Drawing ─────────────────────────────────────────────────────────────────

function drawDetections(dets) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const scale = Math.max(1, overlay.width / 640);
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
}

// ── Pause / Resume (iOS lifecycle) ──────────────────────────────────────────

function pauseAll() {
  console.log("⏸ pause — releasing camera");
  running = false;
  loopBusy = false;
  stopCamera();                   // FREE the hardware so iOS doesn't kill us
}

async function resumeAll() {
  console.log("▶ resume — restarting camera + loop");
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

// The KEY handler: when iOS hides the page (notification, control centre,
// tab switch, lock screen), we immediately release the camera.  When the
// page becomes visible again, we re-acquire it.
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    pauseAll();
  } else {
    resumeAll();
  }
});

// bfcache restore (Safari back/forward cache)
window.addEventListener("pageshow", (e) => {
  if (e.persisted) {
    console.log("pageshow (bfcache restore)");
    resumeAll();
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
// On iOS: the old SW (with skipWaiting + clients.claim) may still be active
// from previous visits.  It can trigger page reloads at any time.
// We must UNREGISTER it and nuke its caches to stop the reload loop.
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
