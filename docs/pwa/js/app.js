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

// Smoothed FPS (exponential moving average)
let smoothFps = 0;
const FPS_ALPHA = 0.15;

// ── Camera ──────────────────────────────────────────────────────────────────

async function startCamera() {
  statusEl.textContent = "Starting camera\u2026";
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "environment",
      width:  { ideal: 1280 },
      height: { ideal: 720 },
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

async function loop() {
  if (!running) return;
  const t0 = performance.now();

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

  requestAnimationFrame(loop);
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
  smoothFps = 0;
  statusEl.textContent = "Loading model\u2026";
  const info = await loadModel(modelSel.value);
  updateHudModel(info);
  statusEl.textContent = "";
  running = true;
  requestAnimationFrame(loop);
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
    requestAnimationFrame(loop);
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
