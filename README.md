# embed-ai — YOLO v26 · Export · Benchmark · PWA

---

## 🎯 Slide 1 — Why Reduce a Model?

```
  ┌─────────────────────────────────────────────────────────────┐
  │                 Cloud / Desktop                             │
  │            GPU · 24 GB VRAM · unlimited power               │
  │                     FP32 model                              │
  └────────────────────────┬────────────────────────────────────┘
                           │  Too big for edge
                           ▼
  ┌─────────────────────────────────────────────────────────────┐
  │              Edge / Mobile / Browser                        │
  │          limited RAM · battery · no GPU                     │
  │              Needs: small · fast · "good enough"            │
  └─────────────────────────────────────────────────────────────┘
```

**Tradeoff triangle:**

```
            Accuracy
               ▲
              ╱ ╲
             ╱   ╲         Pick 2 of 3.
            ╱     ╲        Quantization = sacrifice
           ╱       ╲       accuracy for size + speed.
          ▼─────────▼
       Size ◄─────► Speed
```

---

## 🔧 Slide 2 — The Full Pipeline

```
  yolo26n.pt                          PyTorch weights (FP16 stored, ~5 MB)
       │
       │  model.export(format="tflite")
       ▼
  yolo26n.onnx ◄──────────────────── ONNX intermediate (FP32, ~10 MB)
       │                              Same file → ALL downstream models.
       │
       ├──── onnx2tf (float) ──────────────────────────────────────────┐
       │     yolo26n_float32.tflite  ~10 MB                            │
       │     yolo26n_float16.tflite   ~5 MB                            │
       │                                                               │
       ├──── onnx2tf (int8=True, calibrated on coco128) ──────────────┤
       │     yolo26n_int8.tflite               ~3 MB  ✅ int8_dyn     │
       │     yolo26n_full_integer_quant.tflite  ~3 MB  ⚠ int8_full   │
       │                                                               │
       └──── onnxruntime quantize_static ─────────────────────────────┘
             nodes_to_exclude = /model.23/* (detection head)
             calibrated on output/calibration_frames/
             yolo26n_int8.onnx              ~3.5 MB  ✅ onnx_int8

  ┌─ docs/pwa/models/ ───────────────────────────────────────────────────┐
  │  yolo26n_fp32.onnx   ← copy of yolo26n.onnx        (full FP32)      │
  │  yolo26n_int8.onnx   ← backbone INT8, head FP32    (mixed quant)    │
  └──────────────────────────────────────────────────────────────────────┘
```

---

## 📉 Slide 3 — Size Reduction Chain

```
   Format          Storage     Size       Reduction
  ──────────────  ─────────  ─────────  ────────────
   .pt             FP16       ~5 MB      baseline (stored as half)
   .onnx           FP32      ~10 MB      ⚠ bigger (upcast to float32)
   _float32.tflite FP32      ~10 MB      same weights, TFLite container
   _float16.tflite FP16       ~5 MB      ━━━━━━━━━ 50% of FP32
   _int8.tflite    INT8       ~3 MB      ━━━━━ 30% of FP32 (dynamic)
   _full_int8      INT8       ~3 MB      ━━━━━ 30% of FP32 (full)
```

> The `.pt` stores weights in **FP16** (~5 MB). ONNX/TFLite FP32 **upcast
> to float32** (~10 MB). The real reduction comes from **quantization**:
> FP32 (10 MB) → FP16 (5 MB) → INT8 (3 MB).

---

## 📊 Slide 4 — All Model Variants

```
  Tag        File                          Quant                   Works?
  ─────────  ────────────────────────────  ──────────────────────  ──────
  pt         yolo26n.pt                    FP16 (PyTorch)          yes
  fp32       yolo26n_float32.tflite        FP32 full precision     yes
  fp16       yolo26n_float16.tflite        FP16 weights half       yes
  int8_dyn   yolo26n_int8.tflite           Weights INT8 only       yes
  int8_full  yolo26n_full_integer_quant    EVERYTHING INT8         NO (*)
  onnx_fp32  yolo26n.onnx                  FP32 (PWA)              yes
  onnx_int8  yolo26n_int8.onnx             Mixed: body INT8,       yes
                                           head FP32 (PWA)

  (*) kept in pipeline as demo of what breaks without QAT
```

---

## 🧠 Slide 5 — Why int8_full Breaks (and How We Fix It)

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │ BACKBONE  (model.0 -> model.22)                                     │
  │ Feature extraction: Conv, BatchNorm, activations                    │
  │ Values span wide ranges -> rounding to 256 int8 levels is fine.     │
  │                                                                     │
  │ --> Quantizes well to INT8.                                         │
  └────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   v
  ┌─────────────────────────────────────────────────────────────────────┐
  │ DETECTION HEAD  (model.23)                                          │
  │ Boxes (cv2) + Classes (cv3) + NMS (TopK, Sigmoid, Gather)          │
  │ Outputs: confidence scores 0.30 -- 0.90                             │
  │                                                                     │
  │ INT8 = 256 levels for [0,1] -> step = 0.004                        │
  │ conf 0.34 and conf 0.30 round to the SAME int8 value               │
  │ --> detections below threshold --> objects vanish                    │
  └─────────────────────────────────────────────────────────────────────┘
```

**Three strategies compared:**

```
  Strategy        Backbone    Head          Size     Detects?
  ──────────────  ──────────  ──────────    ───────  ────────
  int8_dyn        INT8 (w)    FP32 (act)    ~3 MB    yes
  int8_full       INT8        INT8          ~3 MB    NO
  onnx_int8       INT8        FP32          ~3.5 MB  yes
  (QAT)           INT8        INT8          ~3 MB    yes (needs retraining)
```

> **int8_dyn** stores weights as int8 but dequantizes to FP32 at
> runtime.  Same file size as int8_full, detection works.
>
> **int8_full** forces everything to int8 -- the detection head
> confidence scores are crushed.  **Kept as a demo of what goes
> wrong** when you blindly quantize without QAT.
>
> **onnx_int8** (used in the PWA) explicitly excludes `/model.23/*`
> from quantization via `nodes_to_exclude`.  Backbone is calibrated
> INT8 (using output/calibration_frames/), head stays FP32.
>
> **QAT** (Quantization-Aware Training) retrains the model with
> quantization in the loop so it learns to produce robust
> confidences despite int8 rounding.  Out of scope here.

---

## ⚡ Slide 6 — Caching: Nothing Is Redone

Every step checks if its output already exists before running:

```
  Step                        Cache check                     Skip message
  ────────────────────────── ─────────────────────────────── ──────────────────────
  1. Download .pt             output/yolo26n.pt exists?        "Weights found"
  2. Calibration frames       calibration_frames/*.jpg count   "already present — skip"
  3. TFLite float export      _float32 + _float16 exist?       "skipping export"
  4. TFLite int8 export       _int8 + _integer + _full exist?  "skipping export"
  5. ONNX int8                _int8.onnx exists?               "already present"
  6. Evaluation (mAP)         eval_cache.json exists?          "loaded from cache"
  7. Benchmark (videos)       benchmark_cache.json exists?     "loaded from cache"
  8. PWA model copy           same size already in docs/?      "already up-to-date"
  9. summary.json             (always rewritten — instant)
```

> **To force a full re-run:** delete `output/` and re-run `python main.py`.
> **To re-run only benchmarks:** delete `output/benchmark_cache.json`.
> **To re-run only evaluation:** delete `output/eval_cache.json`.

---

## 🔬 Slide 7 — Going Bigger or Segmentation?

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
  Faster, smaller model              More visual, ~2x heavier
  Best for speed/size demo           Best for "wow" effect
```

> **For this project:** `yolo26n` detection is ideal.
> - Smallest model → biggest size/speed contrast in the demo.
> - INT8 dynamic quantization works well (3 MB, good FPS).
> - Segmentation would be more visual but ~2× heavier, slower,
>   and the "tiny model on a phone" narrative is weaker.
>
> **If INT8 full worked poorly with `n`**, a bigger model (`yolo26s`)
> would tolerate aggressive quantization better — the detection head
> has more capacity and the confidence distribution is wider.  But the
> demo loses its "micro model" appeal.

---

## 🚀 Quick Start

```bash
pipenv install
pipenv run python main.py
```

### CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `yolo26n` | Model name (auto-downloaded if absent) |
| `--output` | `output` | Directory for all generated artifacts |
| `--summary` | `summary.json` | Summary JSON filename |

Videos are auto-discovered from `resources/*.mp4`.

---

## 📁 Generated Artifacts

```
output/
  yolo26n.pt                             PyTorch weights       (~5 MB)
  yolo26n.onnx                           ONNX FP32 → PWA      (~10 MB)
  yolo26n_int8.onnx                      ONNX mixed → PWA     (~3.5 MB)
  yolo26n_float32.tflite                 TFLite FP32           (~10 MB)
  yolo26n_float16.tflite                 TFLite FP16           (~5 MB)
  yolo26n_int8.tflite                    TFLite int8 dynamic   (~3 MB) ✅
  yolo26n_full_integer_quant.tflite      TFLite full int8      (~3 MB) ⚠
  calibration_frames/                    ~1000 extracted frames
  calibration_frames.yaml                Dataset YAML (nc: 80)
  eval_cache.json                        mAP results cache
  benchmark_cache.json                   FPS results cache
  runs/<tag>/                            Ultralytics val outputs per model
  *_predicted_*.mp4                      Annotated output videos (≤300 frames)
  summary.json                           Consolidated results
```

---

## 📱 PWA — Live Detection on Phone

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

| Model | Size | Quantization |
|-------|------|--------------|
| **FP32** | ~10 MB | Full precision — best accuracy |
| **INT8** | ~3.5 MB | Backbone INT8, detection head FP32 — slightly less accurate |

Both models are **pipeline artifacts** (not downloaded separately):
- FP32 = `yolo26n.onnx`, the ONNX intermediate from TFLite export
- INT8 = `yolo26n_int8.onnx`, mixed-precision: backbone statically
  quantized INT8 (calibrated), detection head `/model.23/*` kept FP32

---

## 🚢 Deploy

### Test locally (before pushing)

```bash
# Serve the PWA over HTTPS on your LAN (camera needs HTTPS)
python serve_local.py

# Output:
#   PWA local server running
#   https://192.168.1.42:8443/pwa/
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

# 4. Done! Share the QR code above.
```
