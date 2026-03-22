import math
import os
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from config import logger, TFLITE_IMG_SIZE, BENCH_MIN_DURATION_S, file_size_mb


def _build_video_writer(video_path: Path, fallback_fps: float = 30.0):
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Cannot read video dimensions from {video_path}")
    if fps <= 0:
        fps = fallback_fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    return width, height, fps, fourcc, total


def _video_output_name(source: str, model_stem: str, device: str) -> str:
    # ex: bad_road_v1_predicted_yolo26n_float32_tflite_cpu.mp4
    return f"{Path(source).stem}_predicted_{model_stem}_{device}.mp4"


def _benchmark_video_mode(
    mode_name: str,
    model_path: str,
    source: str,
    output_video_path: Path,
    imgsz: int,
    device,
    max_frames: int = 0,
    task: str = None,
) -> dict:
    logger.info(
        "[%s] model=%s  device=%s  imgsz=%d", mode_name, model_path, device, imgsz
    )
    if task is not None:
        model = YOLO(model_path, task=task)
    else:
        model_base = os.path.basename(model_path)
        if (
            model_base.endswith("-seg.tflite")
            or model_base.endswith("-seg.pt")
            or model_base.endswith("-seg.onnx")
            or model_base.endswith("-seg")
        ):
            task_type = "segment"
        else:
            task_type = "detect"
        model = YOLO(model_path, task=task_type)
    source_path = Path(source)
    width, height, fps, fourcc, total_in_file = _build_video_writer(source_path)

    effective_total = (
        min(total_in_file, max_frames) if max_frames > 0 else total_in_file
    )
    if 0 < max_frames < total_in_file:
        logger.info(
            "[%s] Video %dx%d@%.0ffps  %d total (capped to %d)",
            mode_name,
            width,
            height,
            fps,
            total_in_file,
            max_frames,
        )
    else:
        logger.info(
            "[%s] Video %dx%d@%.0ffps  %d total",
            mode_name,
            width,
            height,
            fps,
            total_in_file,
        )

    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    cap = cv2.VideoCapture(str(source_path))

    predict_kw: dict = {"imgsz": imgsz, "verbose": False}
    if device is not None:
        predict_kw["device"] = device

    n = 0
    total_conf = 0.0
    det_count = 0
    log_every = max(1, effective_total // 10) if effective_total > 0 else 50

    t0 = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = model.predict(source=frame, **predict_kw)[0]
            writer.write(res.plot())
            n += 1
            if res.boxes is not None and res.boxes.conf is not None:
                c = res.boxes.conf.tolist()
                total_conf += sum(c)
                det_count += len(c)
            if n % log_every == 0 or n == 1:
                el = time.perf_counter() - t0
                logger.info(
                    "[%s] frame %d/%d (%.0f%%) – %.1f fps",
                    mode_name,
                    n,
                    effective_total,
                    (n / effective_total * 100) if effective_total > 0 else 0,
                    n / el if el > 0 else 0,
                )
            if 0 < max_frames <= n:
                break
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - t0
    fps_out = n / elapsed if elapsed > 0 else 0
    avg_conf = total_conf / det_count if det_count > 0 else None
    model_sz = file_size_mb(Path(model_path))

    logger.info(
        "[%s] Done: %d frames %.2fs (%.1f fps) conf=%.4f",
        mode_name,
        n,
        elapsed,
        fps_out,
        avg_conf or 0,
    )

    return {
        "mode": mode_name,
        "model_path": model_path,
        "model_size_mb": model_sz,
        "device": str(device),
        "imgsz": imgsz,
        "frames": n,
        "elapsed_s": round(elapsed, 3),
        "fps": round(fps_out, 2),
        "avg_ms_per_frame": round(elapsed * 1000 / n, 2) if n else None,
        "avg_confidence": round(avg_conf, 4) if avg_conf is not None else None,
        "output_video": str(output_video_path),
    }


def bench(
    model_list: dict[str, str],
    source_paths: list[Path],
    output_root: Path,
    torch_gpu_ok: bool,
    task: str = None,
) -> dict[str, list[dict]]:
    pt_gpu_device = 0 if torch_gpu_ok else "cpu"
    all_video_results: dict[str, list[dict]] = {}

    for source_path in source_paths:
        source = str(source_path)
        video_image_size = TFLITE_IMG_SIZE

        # Read FPS to compute the frame budget for the minimum duration.
        _w, _h, src_fps, _fourcc, src_total = _build_video_writer(source_path)
        max_frames = min(src_total, math.ceil(src_fps * BENCH_MIN_DURATION_S))
        logger.info(
            "▶ Benchmarking: %s  (%.0f fps × %ds = %d frames, total %d)",
            source_path.name,
            src_fps,
            BENCH_MIN_DURATION_S,
            max_frames,
            src_total,
        )

        modes = [
            ("pytorch_gpu", model_list["pt"], video_image_size, pt_gpu_device),
            ("pytorch_cpu", model_list["pt"], video_image_size, "cpu"),
        ]
        # Only add TFLite modes whose files actually exist.
        for tag, mode_name, device in [
            ("fp32", "tflite_fp32_cpu", "cpu"),
            ("fp16", "tflite_fp16_cpu", "cpu"),
            ("int8_dyn", "tflite_int8_dyn_cpu", "cpu"),
            ("int8_full", "tflite_int8_full_cpu", "cpu"),
        ]:
            p = model_list.get(tag)
            if p and Path(p).exists():
                modes.append((mode_name, p, video_image_size, device))
            else:
                logger.warning("Model %s not found at %s -- skipping benchmark", tag, p)

        vid_results: list[dict] = []
        for i, (name, mpath, sz, dev) in enumerate(modes, 1):
            logger.info(
                "--- [%s] Mode %d/%d: %s ---", source_path.name, i, len(modes), name
            )
            model_stem = Path(mpath).stem
            video_name = _video_output_name(source, model_stem, dev)
            vid_results.append(
                _benchmark_video_mode(
                    mode_name=name,
                    model_path=mpath,
                    source=source,
                    output_video_path=output_root / video_name,
                    imgsz=sz,
                    device=dev,
                    max_frames=max_frames,
                    task=task,
                )
            )
        all_video_results[source_path.name] = vid_results
    return all_video_results
