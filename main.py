import argparse
import logging
import os
import sys
import types
import warnings
from pathlib import Path

import torch

# ── Reduce low-value logs during export/conversion ──────────────────────────
# Level 3 = FATAL only; suppresses harmless cuDNN/cuBLAS factory and
# computation-placer duplicate-registration noise from TF/ABSL.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")  # WARNING+
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # silence oneDNN info

# ── LiteRT shim ─────────────────────────────────────────────────────────────
# Ultralytics autobackend tries `from tflite_runtime.interpreter import
# Interpreter`; if that fails it falls back to `import tensorflow` (noisy).
# We register ai_edge_litert *as* tflite_runtime to avoid touching TF at
# inference time entirely.
try:
    from ai_edge_litert import interpreter as _litert_interp

    _shim = types.ModuleType("tflite_runtime")
    _shim.interpreter = _litert_interp
    sys.modules.setdefault("tflite_runtime", _shim)
    sys.modules.setdefault("tflite_runtime.interpreter", _litert_interp)
except ImportError:
    pass
# ─────────────────────────────────────────────────────────────────────────────

warnings.filterwarnings(
    "ignore",
    message=r"Exporting aten::index operator of advanced indexing in opset 20.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"Labels are missing or empty.*training may not work correctly.*",
)

# Import config *after* env vars and shim are set up, but *before* any
# module that uses TF/YOLO.
from config import logger  # noqa: E402

import benchmark  # noqa: E402
import calibration  # noqa: E402
import evaluation  # noqa: E402
import export  # noqa: E402
import summary  # noqa: E402


def _resolve_source(source: str) -> str:
    if Path(source).exists():
        return source
    raise FileNotFoundError(f"Video source not found: {source}")


def run(model: str, output_dir: str, summary_file: str):
    """Export, calibrate, evaluate mAP, and benchmark all videos."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    torch_gpu_ok = torch.cuda.is_available()

    # ── Auto-discover videos ─────────────────────────────────────────
    sources = sorted(str(p) for p in Path("resources").glob("*.mp4"))
    if not sources:
        raise FileNotFoundError("No video sources found in resources/")

    source_paths = [Path(_resolve_source(s)) for s in sources]

    logger.info("=" * 95)
    logger.info("FULL BENCHMARK -- %d video(s)", len(source_paths))
    for s in source_paths:
        logger.info("  - %s", s)
    logger.info("=" * 95)

    # ── Export the model (.pt download) ──────────────────────────────
    weights_pt = export.export(model, output_root)

    # ── Calibration frames from domain videos ────────────────────────
    cal_yaml, cal_diagnostic = calibration.calibrate(
        weights_pt=weights_pt,
        output_root=output_root,
        source_paths=source_paths,
    )

    # ── Build TFLite variants ────────────────────────────────────────
    model_list = export.export_tflite(
        model, output_root, calibration_yaml=cal_yaml,
    )

    # ── Model accuracy evaluation (mAP) ──────────────────────────────
    eval_results = evaluation.evaluate(model_list=model_list)

    # ── Video benchmarks (all sources x all modes) ───────────────────
    all_video_results = benchmark.bench(
        model_list=model_list,
        source_paths=source_paths,
        output_root=output_root,
        torch_gpu_ok=torch_gpu_ok,
    )

    # ── Summary ──────────────────────────────────────────────────────
    return summary.summarize(
        model_list=model_list,
        source_paths=source_paths,
        output_root=output_root,
        summary_file=summary_file,
        cal_yaml=cal_yaml,
        cal_diag=cal_diagnostic,
        eval_results=eval_results,
        all_video_results=all_video_results,
        torch_gpu_ok=torch_gpu_ok,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="YOLO multi-backend video benchmark")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--model", default="yolo26n", help="Model name")
    parser.add_argument("--summary", default="summary.json", help="Summary JSON filename")
    args = parser.parse_args()

    run(output_dir=args.output, model=args.model, summary_file=args.summary)
