# embed-ai

A hands-on demo:

* take a YOLO v26 segmentation model
* shrink it
* measure what breaks
* run it live on a phone

---

## Why Reduce a Model?

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                    Cloud / Desktop                               │
  │               GPU · 24 GB RAM · unlimited power                  │
  └──────────────────────────┬───────────────────────────────────────┘
                             │  big, slow, polluter
                             ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                  Embedded / Mobile / Browser                     │
  │             limited RAM · battery · no GPU                       │
  └──────────────────────────┬───────────────────────────────────────┘
                             │  small, fast, frugal
                             ▼
```

### Mesure the effort

| Goal             | Metric              | Edge constraint       |
|------------------|---------------------|-----------------------|
| Smaller download | File size (MB)      | network               |
| Lower latency    | Inference time (ms) | Real-time at 30 fps   |
| Less memory      | Peak RAM (MB)       | 2–4 GB shared with OS |
| Less energy      | Joules per frame?   | Battery life          |

### Mesure the effect

```
  ☑ Measure file size        (did it shrink?)
  ☑ Measure mAP50 / mAP50-95 (did accuracy drop?)
  ☑ Measure FPS on video     (did it speed up?)
  ☑ Result check             (are predictions still correct?)
```

[Mean Average Precision (mAP)](https://www.ultralytics.com/glossary/mean-average-precision-map)

The tradeoff triangle — pick two:

```
              Accuracy
                 ▲
                ╱ ╲
               ╱   ╲        Quantization =
              ╱     ╲       sacrifice accuracy
             ╱       ╲
            ▼─────────▼
         Size ◄──────► Latency
```

Two recommended builds for any embedding project:

| Build   | Format    |
|---------|-----------|
| Quality | FP16/FP32 |
| Fast    | INT8      |

---

### Embedded Pipeline

```
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 1  Export                                                    │
  │ Download the model                                                │
  └────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 2  Calibrate                                                 │
  │ Run FP32 model on 80 sample frames → diagnostic report            │
  └────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 3  Convert                                                   │
  │ Under the hood: .pt → ONNX → SavedModel → onnx2tf → .tflite       │
  └────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 5  Evaluate                                                  │
  │ Measure: mAP50, mAP50-95, precision, recall                       │
  │ Analyze confusion matrices                                        │
  └────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 6  Bench                                                     │
  │ For each example × each model variant:                            │
  │   Measure: FPS, avg inference time, avg confidence                │
  └────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Step 7  Summarize                                                 │
  │   sources, artifacts, evaluation,  (FPS),                         │
  │   benchmarks, calibration, GPU info                               │
  └───────────────────────────────────────────────────────────────────┘
```

---

## Yolo26 LiteRT Embedding

### Going Bigger ?

[Yolo tasks and modes](https://docs.ultralytics.com/models/yolo26/#supported-tasks-and-modes)

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

**Segmentation models:**

| Model       | Params | PT size | ONNX FP32 | Best for                     |
|-------------|--------|---------|-----------|------------------------------|
| yolo26n-seg | 2.7 M  | 6 MB    | 11 MB     | demonstrator                 |
| yolo26s-seg | 9.7 M  | 20 MB   | 39 MB     | best int8 accuracy           |
| yolo26m-seg | 21.2 M | 44 MB   | 85 MB     | production accuracy          |
| yolo26l-seg | 47.0 M | 97 MB   | 187 MB    | large objects, high accuracy |
| yolo26x-seg | 99.1 M | 205 MB  | 395 MB    | highest accuracy, very heavy |

```
      *.pt                          PyTorch weights
       │
       │  Ultralytics model.export(format="tflite")
       ▼
     *.onnx ◄──────────────────── ONNX intermediate (FP32)
       │
       ├──── onnx2tf (float) ──────────────────────────────────────────┐
       │     *_float32.tflite  ~10 MB  (full precision)                │
       │     *_float16.tflite   ~5 MB  (half precision)                │
       │                                                               │
       ├──── onnx2tf (int8=True, calibrated on coco128) ───────────────┤
       │     *_int8.tflite                ~3 MB  (dynamic-range)       │
       │     *_full_integer_quant.tflite  ~3 MB  (full int8)           │
       │     *_integer_quant.tflite       ~3 MB  (mixed int8 float32)  │
       └───────────────────────────────────────────────────────────────┘
```

### Optimization

| Technique              | What it does                                               |
|------------------------|------------------------------------------------------------|
| **FP16 quantization**  | Halve weight precision (32→16 bits)                        |
| **Dynamic-range INT8** | Quantize weights to int8, activations stay FP32 at runtime |
| **Full-integer INT8**  | Force everything to int8, no float fallback                |
| **Mixed quantization** | Backbone INT8, detection head FP32                         |
| **Calibration**        | Representative dataset to compute activation ranges        |

**What we do NOT use (but could):**

| Technique                         | Why not                         |
|-----------------------------------|---------------------------------|
| QAT (Quantization-Aware Training) | Needs retraining; out of scope  |
| Pruning / Clustering              | More useful on larger models    |
| Edge TPU / NNAPI delegates        | Desktop demo, not mobile native |
| Split suppressions (WebGL)        | WebGL-compatible ONNX export    |

> **Note:** **Dynamic-range INT8** stores weights as int8 but dequantizes to FP32 at runtime.

> **Note:** **Full-integer INT8** forces everything to int8 : the detection head confidence scores are bad.
**Kept as a demo of what goes wrong** when you blindly quantise without tree analysis.

> **Note:** **Full-integer INT8** is not supported
for segmentation models due to Split operators in the graph (available for detect models).

### Tree analysis : an example

```
  ┌──────────────────────────────────────────────────────────────────┐
  │ BACKBONE  (model.0 → model.22)                                   │
  │ Feature extraction: Conv, BatchNorm, activations                 │
  │ Values span wide ranges → rounding to 256 int8 levels is fine.   │
  └──────────────────────────┬───────────────────────────────────────┘
                             │
                             │ ◄────────────── Quantizes well to INT8
                             ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ DETECTION HEAD  (model.23)                                       │
  │ Boxes (cv2) + Classes (cv3) + NMS (TopK, Sigmoid, Gather)        │
  │ Confidence scores live in a narrow range: 0.30 – 0.90            │
  │                                                                  │
  │ INT8 = 256 levels for [0,1] → step = 0.004                       │
  │ conf 0.34 and conf 0.30 round to the SAME int8 value             │
  └──────────────────────────────────────────────────────────────────┘
```

Detections fall below threshold, so objects vanish.

### Runtime

[LiteRT chooses a banckend](https://ai.google.dev/edge/litert/overview#choose-backend)

It depends on the platform:
- **Web**: WebGPU (best), WebGL (fallback), WASM (universal fallback)
- **Mobile**: NNAPI (Android), CoreML (iOS)
- **Desktop**: WebGPU (best), WebGL (fallback), WASM (universal fallback) with asynchronous execution
- **Edge TPU**: Coral USB Accelerator

---

## Yolo26 Progressive Web App (PWA)

```
      *.pt                          PyTorch weights
       │
       │  Ultralytics model.export(format="onnx")
       │
       ├──── Yolo onnx ──────────────────────────────────────────┐
                 │     fp32.onnx  ~10 MB                         │
                 │     opset 17, not end2end                     │
                 ├──── Onnx runtime ─────────────────────────────┤
                           │     quant.onnx  ~3 MB               │
                           │     backbone uint8 type             │
                           │     detection head float32 type     │
                           └─────────────────────────────────────┘
```

<p align="center">
  <img src="docs/pwa/qr.svg" alt="QR code" width="200" /><br/>
  <a href="https://tigroo.github.io/embed-ai/pwa/">https://tigroo.github.io/embed-ai/pwa/</a>
</p>

[Yolo26 formats](https://docs.ultralytics.com/modes/export/#export-formats)

> **Note:** The **end2end** YOLO export includes TopK / GatherElements / Mod
in the detection head ... NOT supported by WebGL :(.
So we export without end2end.

> **Note:**  **NMS** is done in JavaScript: ~1 ms overhead.

> **Note:** Due to Split operators in the segmentation graph,
full int8 quantisation is not possible for ONNX.

### Runtime

Backend negotiation (fastest → safest):
1. WebGPU   (newest, best perf when supported)
2. WebGL    (mature, broad support)
3. WASM     (universal fallback)


Various results:
* WebGPU to manually activate on browser Firefox: about:config → dom.webgpu.enabled : True
* WebGL works → 2-5× faster than WASM on most devices
* WASM on CPU with XNNPACK

---

## Deployment

```bash
pipenv install
pipenv run python main.py
```

### CLI Arguments

| Flag            | Default        | Description                            |
|-----------------|----------------|----------------------------------------|
| `--output`      | `output`       | Directory for all generated artifacts  |
| `--model`       | `yolo26n-seg`  | Model name (auto-downloaded if absent) |
| `--summary`     | `summary.json` | Summary JSON filename                  |
| `--pwa-only`    | False          | Skip TFLite export, only ONNX for PWA  |
| `--calibration` | None           | Path to calibration YAML (optional)    |

Videos are auto-discovered from `resources/*.mp4`.

### Test locally (before pushing)

```bash
python serve_local.py
```
