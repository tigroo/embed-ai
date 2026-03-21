from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from config import logger, TFLITE_IMG_SIZE
from heartbeat import HeartBeat

_CAL_TARGET_TOTAL = 1000    # target total calibration frames across all videos
_CAL_MIN_TOTAL = 300        # absolute minimum (safety net)
_CAL_MIN_PER_VIDEO = 10     # always extract at least this many per video
_CALIBRATION_FRAMES_DIR = "calibration_frames"
_CALIBRATION_YAML_NAME = "calibration_frames.yaml"
_CALIBRATION_GLOB = "calibration_image_sample_data_*"
_SAVED_MODEL_DIR = "saved_model"


def _diagnose_calibration_set(
    cal_images_dir: Path,
    weights: Path,
    sample_limit: int = 80,
    conf: float = 0.25,
) -> dict:
    """Run FP32 model on calibration images and report class distribution.

    INT8 quantization computes per-layer activation ranges from the
    calibration set.  If a class (e.g. *person*) is absent or rare in
    that set, the quantized model will have poor precision on it -- the
    ranges will be optimised for backgrounds / other classes instead.

    This function samples up to *sample_limit* calibration images, runs
    the FP32 model, and returns a detection census.  Warnings are logged
    for common COCO classes that are expected but missing.
    """
    images = sorted(cal_images_dir.glob("*.jpg"))
    if not images:
        logger.warning("No calibration images found in %s", cal_images_dir)
        return {"images_sampled": 0, "total_detections": 0, "class_counts": {}}

    # Subsample for speed (uniform step)
    if len(images) > sample_limit:
        step = len(images) // sample_limit
        images = images[::step][:sample_limit]

    logger.info("Diagnosing calibration set: %d images with FP32 model ...", len(images))
    model = YOLO(str(weights))

    class_counts: dict[str, int] = {}
    images_with_detections = 0

    with HeartBeat("calibration diagnosis", interval=15):
        for img_path in images:
            results = model.predict(
                source=str(img_path), imgsz=TFLITE_IMG_SIZE,
                verbose=False, conf=conf,
            )
            if results and results[0].boxes is not None and len(results[0].boxes):
                images_with_detections += 1
                for cls_id in results[0].boxes.cls.tolist():
                    name = model.names[int(cls_id)]
                    class_counts[name] = class_counts.get(name, 0) + 1

    total_det = sum(class_counts.values())
    sorted_counts = dict(sorted(class_counts.items(), key=lambda x: -x[1]))

    logger.info(
        "Calibration diagnosis: %d images sampled, %d with detections, %d total objects",
        len(images), images_with_detections, total_det,
    )
    for cls_name, cnt in sorted_counts.items():
        logger.info("  %-20s %5d detections", cls_name, cnt)

    # NOTE: this set is DIAGNOSTIC ONLY -- it does NOT influence what the
    # model detects.  COCO covers vehicles, pedestrians, traffic lights,
    # stop signs, but NOT traffic cones, road panels, markings, barriers.
    expected_common = {
        # Vulnerable road users
        "person", "bicycle", "motorcycle",
        # Vehicles
        "car", "bus", "truck",
        # Road infrastructure (COCO subset)
        "traffic light", "stop sign",
    }
    known_names = set(model.names.values())
    missing = (expected_common & known_names) - set(class_counts.keys())
    if missing:
        logger.warning(
            "Classes missing from calibration set: %s -- "
            "the INT8 model may have poor accuracy on these objects. "
            "Add videos containing these classes or increase --cal-frames.",
            ", ".join(sorted(missing)),
        )

    empty_pct = 100 * (1 - images_with_detections / len(images)) if images else 0
    if empty_pct > 60:
        logger.warning(
            "%.0f%% of calibration frames contain no detections -- "
            "too many empty frames dilute activation ranges. "
            "Try videos with more visible objects.",
            empty_pct,
        )

    return {
        "images_sampled": len(images),
        "images_with_detections": images_with_detections,
        "total_detections": total_det,
        "class_counts": sorted_counts,
        "missing_expected_classes": sorted(missing) if missing else [],
    }


def _read_model_names(output_dir: Path, weights: Path | None = None) -> dict[int, str]:
    """Read COCO class names from model metadata, PT checkpoint, or fallback.

    Resolution order:
      1. ``output_dir/saved_model/metadata.yaml`` (post-export)
      2. Load the PT checkpoint and read ``model.names`` (always available)
      3. Generic ``{0: "object"}`` as last resort
    """
    meta_path = output_dir / _SAVED_MODEL_DIR / "metadata.yaml"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        names = meta.get("names", {0: "object"})
        if isinstance(names, list):
            return {i: n for i, n in enumerate(names)}
        return names

    # Fallback: read names from the PT checkpoint itself (always 80 COCO
    # classes for pretrained models) so the calibration YAML is correct
    # even before export.
    if weights is not None and weights.exists():
        try:
            model = YOLO(str(weights))
            if hasattr(model, "names") and model.names:
                logger.info(
                    "Read %d class names from PT model (metadata.yaml not yet created)",
                    len(model.names),
                )
                return dict(model.names)
        except Exception as exc:
            logger.warning("Could not read class names from %s: %s", weights, exc)

    return {0: "object"}


def _extract_calibration_frames(
    video_paths: list[Path],
    output_dir: Path,
    weights: Path | None = None,
) -> str:
    """Extract evenly-spaced frames from domain videos for INT8 calibration.

    The number of frames per video is **proportional to its frame count**
    so that longer videos contribute more calibration diversity.  The
    target total is ``_CAL_TARGET_TOTAL`` (default 1000), with at least
    ``_CAL_MIN_PER_VIDEO`` frames from every video.

    Returns the path (as string) to the generated Ultralytics-compatible YAML.
    """
    cal_root = output_dir / _CALIBRATION_FRAMES_DIR
    cal_images = cal_root / "images"
    cal_labels = cal_root / "labels"
    cal_yaml = output_dir / _CALIBRATION_YAML_NAME

    # ── Read frame counts from every video ───────────────────────────
    video_frame_counts: list[tuple[Path, int]] = []
    for vp in video_paths:
        cap = cv2.VideoCapture(str(vp))
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fc <= 0:
            logger.warning("Cannot read frame count from %s -- skipping", vp)
            continue
        video_frame_counts.append((vp, fc))

    if not video_frame_counts:
        logger.error("No readable videos -- cannot build calibration set")
        cal_yaml.write_text("")
        return str(cal_yaml)

    total_video_frames = sum(fc for _, fc in video_frame_counts)
    target = max(_CAL_MIN_TOTAL, _CAL_TARGET_TOTAL)

    # Proportional allocation with per-video minimum
    allocations: list[tuple[Path, int, int]] = []  # (path, frame_count, n_extract)
    for vp, fc in video_frame_counts:
        n = max(_CAL_MIN_PER_VIDEO, round(fc / total_video_frames * target))
        n = min(n, fc)  # cannot extract more frames than the video has
        allocations.append((vp, fc, n))

    expected_total = sum(n for _, _, n in allocations)

    # ── Check if frames are already extracted ────────────────────────
    if cal_yaml.exists():
        n_existing = len(list(cal_images.glob("*.jpg")))
        if n_existing >= expected_total:
            logger.info(
                "Calibration frames already present (%d >= %d) -- skip extraction",
                n_existing, expected_total,
            )
            return str(cal_yaml)

    cal_images.mkdir(parents=True, exist_ok=True)
    cal_labels.mkdir(parents=True, exist_ok=True)

    # ── Extract ──────────────────────────────────────────────────────
    total_saved = 0
    for vp, fc, n_extract in allocations:
        cap = cv2.VideoCapture(str(vp))
        step = max(1, fc // n_extract)
        saved = 0
        for idx in range(0, fc, step):
            if saved >= n_extract:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                stem = f"{vp.stem}_{idx:06d}"
                cv2.imwrite(str(cal_images / f"{stem}.jpg"), frame)
                (cal_labels / f"{stem}.txt").touch()
                saved += 1
        cap.release()
        total_saved += saved
        logger.info(
            "Extracted %d calibration frames from %s (%d total video frames)",
            saved, vp.name, fc,
        )

    # Write Ultralytics-compatible dataset YAML
    names = _read_model_names(output_dir, weights=weights)
    cal_data = {
        "path": str(cal_root.resolve()),
        "train": "images",
        "val": "images",
        "nc": len(names),
        "names": names,
    }
    with open(cal_yaml, "w") as f:
        yaml.dump(cal_data, f, default_flow_style=False)

    logger.info("Calibration dataset ready: %d images -> %s", total_saved, cal_yaml)
    return str(cal_yaml)


def calibrate(
    weights_pt: Path,
    output_root: Path,
    source_paths: list[Path],
) -> tuple[str, dict]:
    """Prepare a calibration dataset for INT8 quantization and diagnose quality."""
    cal_yaml = _extract_calibration_frames(source_paths, output_root, weights=weights_pt)

    cal_diagnostic = _diagnose_calibration_set(
        output_root / _CALIBRATION_FRAMES_DIR / "images",
        weights_pt,
    )

    return cal_yaml, cal_diagnostic
