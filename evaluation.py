from pathlib import Path

from ultralytics import YOLO

from config import logger, TFLITE_IMG_SIZE, file_size_mb

_EVAL_DATA = "coco8.yaml"


def evaluate(model_list: dict[str, str], runs_dir: Path | None = None) -> list[dict]:
    """Evaluate exported models on the given dataset and return metrics.

    If *runs_dir* is given, Ultralytics validation artifacts (confusion
    matrices, sample predictions) are stored under ``runs_dir/<tag>/``
    instead of the default ``runs/detect/valN``.
    """
    results: list[dict] = []
    for tag, model_path in model_list.items():
        p = Path(model_path)
        if not p.exists():
            logger.warning("Model %s not found -- skipping eval", model_path)
            continue
        logger.info("Evaluating %-10s (%s) on %s ...", tag, p.name, _EVAL_DATA)
        model = YOLO(str(p), task="detect")
        val_kw: dict = {
            "data": _EVAL_DATA,
            "imgsz": TFLITE_IMG_SIZE,
            "verbose": False,
        }
        if runs_dir is not None:
            val_kw["project"] = str(runs_dir.resolve())
            val_kw["name"] = tag
            val_kw["exist_ok"] = True
        try:
            metrics = model.val(**val_kw)
            entry = {
                "tag": tag,
                "model_path": str(p),
                "model_size_mb": file_size_mb(p),
                "mAP50": round(float(metrics.box.map50), 4),
                "mAP50_95": round(float(metrics.box.map), 4),
                "precision": round(float(metrics.box.mp), 4),
                "recall": round(float(metrics.box.mr), 4),
            }
        except Exception as e:
            logger.error("Eval failed for %s: %s", tag, e)
            entry = {
                "tag": tag,
                "model_path": str(p),
                "model_size_mb": file_size_mb(p),
                "mAP50": None,
                "mAP50_95": None,
                "precision": None,
                "recall": None,
                "error": str(e),
            }
        results.append(entry)
        if entry.get("mAP50") is not None:
            logger.info(
                "  -> mAP50=%.4f  mAP50-95=%.4f  P=%.4f  R=%.4f",
                entry["mAP50"], entry["mAP50_95"],
                entry["precision"], entry["recall"],
            )
    return results
