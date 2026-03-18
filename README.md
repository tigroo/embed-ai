# embed_ai video benchmark

Benchmarks YOLO26n detection across **5 backends** on all personal videos:

| Mode | Model | Device |
|------|-------|--------|
| PyTorch GPU | `yolo26n.pt` | CUDA (if available) |
| PyTorch CPU | `yolo26n.pt` | CPU |
| TFLite fp32 | `yolo26n_float32.tflite` | CPU XNNPACK |
| TFLite fp16 | `yolo26n_float16.tflite` | CPU XNNPACK |
| TFLite int8 | `yolo26n_integer_quant.tflite` | CPU XNNPACK |

It also:
- **Extracts calibration frames** from your domain videos for better INT8 quantization
- **Evaluates mAP** (mAP50, mAP50-95, precision, recall) on COCO8 for every variant
- Writes annotated videos + a consolidated JSON summary

## Quick start

```bash
pipenv install
pipenv run python main.py                     # all 3 videos in resources/
```

## Fast smoke run (first 30 frames per video)

```bash
pipenv run python main.py --max-frames 30
```

## Single video only

```bash
pipenv run python main.py --source resources/bad_road_v1.mp4
```

## Force re-export + re-calibrate INT8

```bash
pipenv run python main.py --force-export --recalibrate
```

## Skip mAP evaluation (faster)

```bash
pipenv run python main.py --no-eval
```

## Calibration tuning

```bash
# More frames = better calibration accuracy (default: 20 per video)
pipenv run python main.py --recalibrate --cal-frames 50 --force-export
```

## Key CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--sources` | all `.mp4` in `resources/` | Video paths to benchmark |
| `--source` | — | Single video shortcut |
| `--eval` / `--no-eval` | `--eval` | Run mAP evaluation on COCO8 |
| `--eval-data` | `coco8.yaml` | Dataset YAML for mAP eval |
| `--recalibrate` | off | Re-extract calibration frames |
| `--cal-frames` | 20 | Frames per video for calibration |
| `--force-export` | off | Force TFLite re-export |
| `--max-frames` | all | Limit frames per benchmark pass |
| `--output-dir` | `output` | Output directory |

## Outputs

- `output/calibration_frames/` — extracted domain images for INT8 calibration
- `output/calibration_frames.yaml` — Ultralytics dataset YAML for calibration
- `output/yolo26n_saved_model/` — SavedModel + all TFLite variants
- `output/*_predicted_*.mp4` — annotated videos per (video × mode)
- `output/video_benchmark_summary.json` — full results (eval + timing)

