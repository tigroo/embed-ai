/**
 * Main application — camera, render loop, HUD, UI controls.
 */
import { loadModel, detect } from "./detector.js";
import { COCO_LABELS, classColor } from "./coco-labels.js";

// ── DOM refs ────────────────────────────────────────────────────────────────
const video    = document.getElementById("camera");
const overlay  = document.getElementById("overlay");
const ctx      = overlay.getContext("2d");
const statusEl = document.getElementById("status");
const modelSel = document.getElementById("model-select");
const confIn   = document.getElementById("conf-slider");
const confVal  = document.getElementById("conf-value");

// HUD elements
const hFps   = document.getElementById("h-fps");
const hInf   = document.getElementById("h-inf");
const hObj   = document.getElementById("h-obj");
const hModel = document.getElementById("h-model");
const hSize  = document.getElementById("h-size");
const hBack  = document.getElementById("h-back");

let running = false;
let confThreshold = 0.35;
let loopBusy = false;          // prevent overlapping inferences
let consecutiveErrors = 0;     // track silent failures
const MAX_CONSECUTIVE_ERR = 5; // stop loop after N consecutive errors

// Smoothed FPS (exponential moving average)
let smoothFps = 0;
const FPS_ALPHA = 0.15;

// Minimum gap (ms) between inference starts — lets the GC breathe on mobile.
// 50 ms ≈ 18 FPS cap; keeps iOS WebKit stable.
const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
const MIN_FRAME_GAP_MS = isMobile ? 80 : 30;

// ── Camera ──────────────────────────────────────────────────────────────────

async function startCamera() {
  statusEl.textContent = "Starting camera\u2026";
  // Lower resolution on mobile: reduces getImageData() cost and memory pressure.
  const camW = isMobile ? 640 : 1280;
  const camH = isMobile ? 480 : 720;
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "environment",
      width:  { ideal: camW },
      height: { ideal: camH },
    },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  overlay.width  = video.videoWidth;
  overlay.height = video.videoHeight;
  statusEl.textContent = "";
}

// ── Render loop ─────────────────────────────────────────────────────────────

function scheduleNext() {
  if (!running) return;
  // Use setTimeout to guarantee a minimum gap so the browser / GC can
  // reclaim memory between frames (critical for iOS WASM).
  setTimeout(() => requestAnimationFrame(loop), MIN_FRAME_GAP_MS);
}

async function loop() {
  if (!running) return;
  if (loopBusy) { scheduleNext(); return; }   // skip if previous frame still running
  loopBusy = true;

  const t0 = performance.now();

  try {
    // Guard: if video track ended (e.g. iOS revoked camera), stop gracefully
    if (video.srcObject) {
      const tracks = video.srcObject.getVideoTracks();
      if (!tracks.length || tracks[0].readyState === "ended") {
        console.warn("Camera track ended — pausing loop");
        running = false;
        loopBusy = false;
        statusEl.textContent = "Camera lost — tap Model selector to restart";
        return;
      }
    }

    const { dets, inferMs } = await detect(video, confThreshold);
    drawDetections(dets);

    const totalMs = performance.now() - t0;
    const instantFps = 1000 / totalMs;
    smoothFps = smoothFps === 0
      ? instantFps
      : smoothFps * (1 - FPS_ALPHA) + instantFps * FPS_ALPHA;

    // Update HUD
    hFps.textContent = smoothFps.toFixed(1);
    hInf.textContent = inferMs.toFixed(0) + " ms";
    hObj.textContent = dets.length;
    consecutiveErrors = 0;
  } catch (err) {
    console.error("Loop error:", err);
    consecutiveErrors++;
    if (consecutiveErrors >= MAX_CONSECUTIVE_ERR) {
      running = false;
      statusEl.textContent = `Stopped (${err.message}). Switch model to retry.`;
      loopBusy = false;
      return;
    }
  }

  loopBusy = false;
  scheduleNext();
}

function drawDetections(dets) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const scale = Math.max(1, overlay.width / 640);

  for (const d of dets) {
    const color = classColor(d.cls);
    const label = `${COCO_LABELS[d.cls] ?? d.cls} ${(d.conf * 100).toFixed(0)}%`;
    const bw = d.x2 - d.x1;
    const bh = d.y2 - d.y1;

    // Bounding box
    ctx.strokeStyle = color;
    ctx.lineWidth = 2 * scale;
    ctx.strokeRect(d.x1, d.y1, bw, bh);

    // Label background
    const fontSize = Math.round(13 * scale);
    ctx.font = `bold ${fontSize}px sans-serif`;
    const tw = ctx.measureText(label).width + 8 * scale;
    const lh = fontSize + 6 * scale;
    ctx.fillStyle = color;
    ctx.fillRect(d.x1, d.y1 - lh, tw, lh);

    // Label text
    ctx.fillStyle = "#fff";
    ctx.fillText(label, d.x1 + 4 * scale, d.y1 - 4 * scale);
  }
}

// ── HUD update after model load ─────────────────────────────────────────────

function updateHudModel(info) {
  hModel.textContent = info.label;
  hSize.textContent  = info.size;
  hBack.textContent  = info.backend;
}

// ── UI events ───────────────────────────────────────────────────────────────

modelSel.addEventListener("change", async () => {
  running = false;
  loopBusy = false;
  smoothFps = 0;
  consecutiveErrors = 0;
  statusEl.textContent = "Loading model\u2026";
  try {
    // Re-check camera is still alive; restart if needed
    if (!video.srcObject || !video.srcObject.getVideoTracks().length
        || video.srcObject.getVideoTracks()[0].readyState === "ended") {
      await startCamera();
    }
    const info = await loadModel(modelSel.value);
    updateHudModel(info);
    statusEl.textContent = "";
    running = true;
    scheduleNext();
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
    console.error("Model switch error:", err);
  }
});

confIn.addEventListener("input", () => {
  confThreshold = parseFloat(confIn.value);
  confVal.textContent = confThreshold.toFixed(2);
});

// ── Boot ────────────────────────────────────────────────────────────────────

async function init() {
  try {
    await startCamera();
    statusEl.textContent = "Loading model\u2026";
    const info = await loadModel(modelSel.value);
    updateHudModel(info);
    statusEl.textContent = "";
    running = true;
    scheduleNext();
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
    console.error(err);
  }
}

init();

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

// Register service worker
if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("sw.js").catch(console.error);
}
