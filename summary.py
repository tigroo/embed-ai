import json
from pathlib import Path

import torch

from config import logger, file_size_mb, TFLITE_IMG_SIZE


def summarize(
    model_list: dict[str, str],
    source_paths: list[Path],
    output_root: Path,
    summary_file: str,
    cal_yaml: str,
    cal_diag: dict,
    eval_results: list[dict],
    all_video_results: dict[str, list[dict]],
    torch_gpu_ok: bool,
) -> dict:
    """Write a consolidated JSON summary and return it."""
    artifacts = {
        tag: {"path": str(p), "size_mb": file_size_mb(Path(p))}
        for tag, p in model_list.items()
    }
    summary = {
        "sources": [str(p) for p in source_paths],
        "imgsz_tflite": TFLITE_IMG_SIZE,
        "torch_cuda": torch_gpu_ok,
        "gpu_name": torch.cuda.get_device_name(0) if torch_gpu_ok else None,
        "calibration_yaml": cal_yaml,
        "calibration_diagnosis": cal_diag,
        "artifacts": artifacts,
        "evaluation": eval_results,
        "videos": dict(all_video_results),
    }
    summary_path = output_root / summary_file
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Summary JSON written to %s", summary_path)
    return summary
