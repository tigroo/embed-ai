import argparse
import json
import logging
import os
import sys
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
from config import logger

import benchmark
import calibration
import evaluation
import export
import summary


def _resolve_source(source: str) -> str:
    if Path(source).exists():
        return source
    raise FileNotFoundError(f"Video source not found: {source}")


def run(
    model: str,
    output_dir: str,
    summary_file: str,
    pwa_only: bool = False,
    calibration_name: str = None,
):
    if calibration_name is None:
        cal_name = "resources"
        cal_yaml_path = None
    else:
        cal_name = calibration_name
        cal_yaml_path = f"{calibration_name}.yaml"

    output_root = Path(output_dir) / f"{model}_{cal_name}"
    output_root.mkdir(parents=True, exist_ok=True)

    if pwa_only:
        export.export_onnx_for_pwa(output_root, model=model)
    else:
        torch_gpu_ok = torch.cuda.is_available()
        sources = sorted(str(p) for p in Path("resources").glob("*.mp4"))
        if not sources:
            raise FileNotFoundError("No video sources found in resources/")
        source_paths = [Path(_resolve_source(s)) for s in sources]
        logger.info("=" * 95)
        logger.info("FULL BENCHMARK -- %d video(s)", len(source_paths))
        for s in source_paths:
            logger.info("  - %s", s)
        logger.info("=" * 95)
        weights_pt = export.export(model, output_root)
        if calibration_name is None:
            cal_yaml, cal_diagnostic = calibration.calibrate(
                weights_pt=weights_pt,
                output_root=output_root,
                source_paths=source_paths,
            )
        else:
            cal_yaml = cal_yaml_path
            cal_diagnostic = None
        # Determine task type from model name
        model_base = os.path.basename(model)
        if model_base.endswith("-seg"):
            task = "segment"
        else:
            task = "detect"
        # Export TFLite variants according to task
        model_list = export.export_tflite(
            model, output_root, calibration_yaml=cal_yaml, task=task
        )
        export.export_onnx_for_pwa(output_root, model=model)
        eval_cache = output_root / "eval_cache.json"
        if eval_cache.exists():
            eval_results = json.loads(eval_cache.read_text())
            logger.info(
                "Evaluation results loaded from cache (%d entries)", len(eval_results)
            )
        else:
            eval_results = evaluation.evaluate(
                model_list=model_list, runs_dir=output_root / "runs"
            )
            eval_cache.write_text(json.dumps(eval_results, indent=2))
            logger.info("Evaluation results cached to %s", eval_cache)
        bench_cache = output_root / "benchmark_cache.json"
        if bench_cache.exists():
            all_video_results = json.loads(bench_cache.read_text())
            logger.info(
                "Benchmark results loaded from cache (%d videos)",
                len(all_video_results),
            )
        else:
            all_video_results = benchmark.bench(
                model_list=model_list,
                source_paths=source_paths,
                output_root=output_root,
                torch_gpu_ok=torch_gpu_ok,
            )
            bench_cache.write_text(json.dumps(all_video_results, indent=2))
            logger.info("Benchmark results cached to %s", bench_cache)
        summary.summarize(
            model_list=model_list,
            source_paths=source_paths,
            output_root=output_root,
            summary_file=f"{cal_name}_{summary_file}",
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
    parser.add_argument(
        "--summary", default="summary.json", help="Summary JSON filename"
    )
    parser.add_argument(
        "--pwa-only", action="store_true", help="Export only models for PWA"
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="Calibration YAML name (ex: coco128). If absent, use local extracted frames from resources/.",
    )
    args = parser.parse_args()

    run(
        model=args.model,
        output_dir=args.output,
        summary_file=args.summary,
        pwa_only=args.pwa_only,
        calibration_name=args.calibration,
    )
