# embed-ai

> A hands-on demo: take a YOLO v26 detection model, shrink it from
> 10 MB to 3 MB, measure what breaks, and run it live on a phone.

---

## Why Reduce a Model?

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                    Cloud / Desktop                               │
  │               GPU · 24 GB VRAM · unlimited power                 │
  │                        FP32 model                                │
  └──────────────────────────┬───────────────────────────────────────┘
                             │  Too big, too slow, too hungry
                             ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                  Edge / Mobile / Browser                         │
  │             limited RAM · battery · no GPU (often)               │
  │                 Needs: small · fast · "good enough"              │
  └──────────────────────────────────────────────────────────────────┘
```

**Four reasons to shrink:**

| Goal                | Metric              | Edge constraint       |
|---------------------|---------------------|-----------------------|
| Smaller download    | File size (MB)      | 5G, app store limits  |
| Lower latency       | Inference time (ms) | Real-time at 30 fps   |
| Less memory         | Peak RAM (MB)       | 2–4 GB shared with OS |
| Less energy         | Joules per frame    | Battery life matters  |

**The tradeoff triangle — pick two:**

```
              Accuracy
                 ▲
                ╱ ╲
               ╱   ╲        Quantization =
              ╱     ╲       sacrifice accuracy
             ╱       ╲      for size + speed.
            ▼─────────▼
         Size ◄──────► Latency
```

**Two recommended builds for any edge project:**

| Build     | Format     | Use case                         |
|-----------|------------|----------------------------------|
| Quality   | FP16/FP32  | Best accuracy, larger, slower    |
| Fast      | INT8       | Smallest, fastest, "good enough" |

---

## Pipeline

```
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 1  export                                                    │
  │ Download / locate yolo26n.pt.                                     │
  └────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 2  calibration                                               │
  │ Run FP32 model on 80 sample frames → diagnostic report.           │
  └────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 3  tflite                                                    │
  │ Under the hood: .pt → ONNX → SavedModel → onnx2tf → .tflite       │
  │    Call 1: format="tflite"                                        │
  │     → yolo26n_float32.tflite  (~10 MB)                            │
  │     → yolo26n_float16.tflite  (~5 MB)                             │
  │   Call 2: format="tflite", int8=True, data="coco128.yaml"         │
  │     → yolo26n_int8.tflite             (~3 MB) — dynamic-range     │
  │     → yolo26n_full_integer_quant.tflite (~3 MB) — full int8       │
  └────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 4  pwa                                                       │
  │ Export ONNX at opset 17, end2end=False, no built-in NMS           │
  │ Quantize backbone to INT8, keep detection head FP32.              │
  │ These models are WebGL-compatible (no GatherElements/Mod/TopK).   │
  └────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 5  evaluation                                                │
  │ Measure: mAP50, mAP50-95, precision, recall.                      │
  │ Analyze confusion matrices.                                       │
  └────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 6  benchmark                                                 │
  │ For each video × each model variant (pt GPU/CPU + 4 TFLite):      │
  │   Measure: FPS, avg inference time, avg confidence.               │
  └────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 7  summary                                                   │
  │   sources, artifacts (sizes), evaluation (mAP), benchmarks (FPS), │
  │   calibration report, GPU info.                                   │
  └───────────────────────────────────────────────────────────────────┘
```

---

## Slide 3 — The Conversion Pipeline in Detail

```
  yolo26n.pt                          PyTorch weights
       │
       │  Ultralytics model.export(format="tflite")
       ▼
  yolo26n.onnx ◄──────────────────── ONNX intermediate (FP32, ~10 MB)
       │                              Upcast: FP16 → FP32 (weights double)
       │
       ├──── onnx2tf (float) ──────────────────────────────────────────┐
       │     yolo26n_float32.tflite  ~10 MB  (full precision)          │
       │     yolo26n_float16.tflite   ~5 MB  (half precision)          │
       │                                                               │
       ├──── onnx2tf (int8=True, calibrated on coco128) ───────────────┤
       │     yolo26n_int8.tflite               ~3 MB  (dynamic-range)  │
       │     yolo26n_full_integer_quant.tflite  ~3 MB  (full int8)     │
       │                                                               │
       └──── Ultralytics export(format="onnx", opset=17) ──────────────┘
             + onnxruntime quantize_dynamic
             → yolo26n_fp32.onnx              ~10 MB  (PWA full)
             → yolo26n_int8.onnx              ~3.5 MB (PWA mixed)
                backbone INT8 / head FP32
                (nodes_to_exclude = /model.23/*)
```

**General LiteRT conversion path:**

```
  Train → Export (ONNX) → Graph optimizations → Convert → .tflite
    → Post-process (FP16 / INT8 / prune) → Validate (mAP) → Deploy
```

---

## Optimization Techniques Used

| Technique              | What it does                                               |
|------------------------|------------------------------------------------------------|
| **FP16 quantization**  | Halve weight precision (32→16 bits)                        |
| **Dynamic-range INT8** | Quantize weights to int8, activations stay FP32 at runtime |
| **Full-integer INT8**  | Force everything to int8, no float fallback                |
| **Mixed quantization** | Backbone INT8, detection head FP32                         |
| **Calibration**        | Representative dataset to compute activation ranges        |

**What we do NOT use (but could):**

| Technique | Why not                                                |
|-----------|--------------------------------------------------------|
| QAT (Quantization-Aware Training) | Needs retraining; out of scope |
| Pruning / Clustering | More useful on larger models                |
| Edge TPU / NNAPI delegates | Desktop demo, not mobile native       |

**Checklist after each optimisation:**

```
  ☑ Measure file size        (did it shrink?)
  ☑ Measure mAP50 / mAP50-95 (did accuracy drop?)
  ☑ Measure FPS on video     (did it speed up?)
  ☑ Visual check             (are detections still correct?)
```

> The real reduction comes from **quantisation within TFLite**:
> FP32 (10 MB) → FP16 (5 MB) → INT8 (3 MB).

> **int8_dyn** stores weights as int8 but dequantizes to FP32 at
> runtime.  Same file size as int8_full, detection works.
>
> **int8_full** forces everything to int8 — the detection head
> confidence scores are crushed.  **Kept as a demo of what goes
> wrong** when you blindly quantise without QAT.
>
> **onnx_int8** (used in the PWA) explicitly excludes `/model.23/*`
> from quantisation via `nodes_to_exclude`.  Backbone is calibrated
> INT8 (using coco128), head stays FP32.
>
> **QAT** (Quantization-Aware Training) retrains the model with
> quantisation in the loop so it learns to produce robust confidences
> despite int8 rounding.  Out of scope here.

---

##  Why int8_full Breaks (and How We Fix It)

```
  ┌──────────────────────────────────────────────────────────────────┐
  │ BACKBONE  (model.0 → model.22)                                   │
  │ Feature extraction: Conv, BatchNorm, activations                 │
  │ Values span wide ranges → rounding to 256 int8 levels is fine.   │
  │                                                                  │
  │ → Quantizes well to INT8.                                        │
  └──────────────────────────┬───────────────────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ DETECTION HEAD  (model.23)                                       │
  │ Boxes (cv2) + Classes (cv3) + NMS (TopK, Sigmoid, Gather)       │
  │ Confidence scores live in a narrow range: 0.30 – 0.90            │
  │                                                                  │
  │ INT8 = 256 levels for [0,1] → step = 0.004                      │
  │ conf 0.34 and conf 0.30 round to the SAME int8 value            │
  │ → detections fall below threshold → objects vanish               │
  └──────────────────────────────────────────────────────────────────┘
```

---

## Slide 8 — Runtime and WebGL Compatibility

**TFLite runtime (desktop benchmark):**

```
  main.py benchmarks use LiteRT (ai-edge-litert), the successor to
  tflite_runtime.  A shim in main.py registers it as tflite_runtime
  so Ultralytics finds it automatically.

  LiteRT delegates used: XNNPACK (CPU, default).
  No GPU delegate / NNAPI in this demo (desktop focus).
```

**ONNX Runtime (PWA / browser):**

```
  Backend negotiation (fastest → safest):
    1. WebGPU   (newest, best perf when supported)
    2. WebGL    (mature, broad support)
    3. WASM     (universal fallback)

  The end2end YOLO export includes TopK / GatherElements / Mod
  in the detection head — NOT supported by WebGL.

  Solution: export with end2end=False (opset 17).
  Output: [1, 84, 8400] raw tensor.
  NMS is done in JavaScript (detector.js): ~1 ms overhead.

  Result: WebGL works → 2-5× faster than WASM on most devices.
```

---

## Slide 10 — Going Bigger or Segmentation?

```
  Model           Params    PT size   ONNX FP32   Best for
  ──────────────  ────────  ────────  ──────────  ──────────────────────
  yolo26n         2.4 M      5 MB     10 MB       ✅ tiny + fast demo
  yolo26s         9.2 M     19 MB     37 MB       better int8 accuracy
  yolo26m        20.0 M     40 MB     79 MB       production accuracy
  yolo26n-seg     2.7 M      6 MB     11 MB       pixel masks (visual!)
```

**Detection (boxes) vs Segmentation (pixel masks):**

```
  Detection                          Segmentation
  ┌────────────────────────┐         ┌────────────────────────┐
  │                        │         │                        │
  │    ┌──────────┐        │         │    ░░░░░░░░░░          │
  │    │ person   │        │         │    ░ person ░          │
  │    │  87%     │        │         │    ░  87%   ░          │
  │    └──────────┘        │         │    ░░░░░░░░░░          │
  │                        │         │                        │
  └────────────────────────┘         └────────────────────────┘
  Faster, smaller model              More visual, ~2× heavier
```

> **For this project:** `yolo26n` detection is ideal.
> Smallest model → biggest size/speed contrast in the demo.

---

## Slide 11 — PWA: Live Detection on a Phone

<p align="center">
  <img src="docs/pwa/qr.svg" alt="QR code" width="200" /><br/>
  <a href="https://tigroo.github.io/embed-ai/pwa/">https://tigroo.github.io/embed-ai/pwa/</a>
</p>

```
  ┌─────────────────────────────────────────────┐
  │          YOLO v26 — Live Detect             │
  │                                             │
  │  ┌───────────────────────────────────────┐  │
  │  │  ┌─────┐                              │  │
  │  │  │ HUD │ FPS: 12.3                    │  │
  │  │  │     │ Inference: 82 ms             │  │
  │  │  │     │ Objects: 3                   │  │
  │  │  │     │ Model: INT8                  │  │
  │  │  │     │ Backend: webgl               │  │
  │  │  └─────┘                              │  │
  │  │         ┌──────────┐                  │  │
  │  │         │ person   │                  │  │
  │  │         │  87%     │                  │  │
  │  │         └──────────┘  ┌─────┐         │  │
  │  │                       │ car │         │  │
  │  │                       │ 72% │         │  │
  │  │                       └─────┘         │  │
  │  └───────────────────────────────────────┘  │
  │                                             │
  │  [Model ▾ INT8 — ~3 MB]  [Conf ━━●━━ 0.35] │
  └─────────────────────────────────────────────┘
```

| Model | Size | Quantization | Backend |
|-------|------|-------------|---------|
| **FP32** | ~10 MB | Full precision | WebGPU / WebGL / WASM |
| **INT8** | ~3.5 MB | Backbone INT8, head FP32 | WebGPU / WebGL / WASM |

Both models are **pipeline artifacts** generated by `main.py` (step 4):
- FP32 = ONNX exported at opset 17, `end2end=False`
- INT8 = same graph, backbone quantized with `onnxruntime.quantize_dynamic`,
  detection head `/model.23/*` excluded to preserve confidence precision

NMS runs in JavaScript (`detector.js`): decode `[1, 84, 8400]` raw
output, per-class greedy NMS with IoU threshold 0.45.

---

## Quick Start

```bash
pipenv install
pipenv run python main.py
```

### CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `yolo26n-seg` | Model name (auto-downloaded if absent) |
| `--output` | `output` | Directory for all generated artifacts |
| `--summary` | `summary.json` | Summary JSON filename |

Videos are auto-discovered from `resources/*.mp4`.

> **Note:** INT8 TFLite export is not available for segmentation models
> due to a bug in Ultralytics 8.4.x (calibration data lacks seg masks).
> The pipeline handles this gracefully — FP32/FP16 TFLite and both PWA
> ONNX models are still produced.  Use `--model yolo26n` (detection) for
> the full INT8 chain.

### Output

```
output/
  summary.json                  ← consolidated results
  yolo26n.pt                    ← PyTorch weights
  yolo26n_float32.tflite        ← FP32
  yolo26n_float16.tflite        ← FP16
  yolo26n_int8.tflite           ← INT8 dynamic-range
  yolo26n_full_integer_quant.tflite ← INT8 full (broken demo)
  eval_cache.json               ← mAP results
  benchmark_cache.json          ← FPS results
  *_predicted_*.mp4             ← annotated videos
  calibration_frames/           ← extracted frames
  runs/                         ← Ultralytics val artifacts

docs/pwa/models/
  yolo26n_fp32.onnx             ← PWA FP32 model
  yolo26n_int8.onnx             ← PWA INT8 model
```

---

## Deploy

### Test locally (before pushing)

```bash
# Serve the PWA over HTTPS on your LAN (camera needs HTTPS)
python serve_local.py

# Output:
#   PWA local server running
#   https://192.168.1.42:8443/pwa/
#   Models: yolo26n_fp32.onnx (9.5 MB), yolo26n_int8.onnx (3.1 MB)
#
#   Open this URL on your phone (same WiFi).
#   Accept the security warning (self-signed cert).
```

### Deploy to GitHub Pages

```bash
# 1. Run the full pipeline
pipenv run python main.py

# 2. Push to GitHub
git add -A && git commit -m "pipeline run" && git push

# 3. Enable GitHub Pages (once)
#    Settings → Pages → Source: branch main, folder /docs

# 4. Share the QR code or URL above.
```
