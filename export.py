"""
YOLO export: download weights and produce TFLite variants.
"""

import os
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

from config import logger, file_size_mb
from heartbeat import HeartBeat


def _hide_tf_gpus():
    """Prevent TensorFlow from using GPUs (avoids XLA/libdevice crash).

    Returns state needed by ``_restore_tf_gpus`` to undo the change.
    TF's ``set_visible_devices`` internally sets ``CUDA_VISIBLE_DEVICES=-1``
    which also hides GPUs from PyTorch.  We save and restore the env var
    explicitly so that subsequent PyTorch calls still see the GPU.
    """
    saved_env = os.environ.get("CUDA_VISIBLE_DEVICES")  # may be None
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.set_visible_devices([], "GPU")
            logger.info("Hid %d TF GPU(s) for export", len(gpus))
        return gpus, saved_env
    except ImportError:
        return [], saved_env


def _restore_tf_gpus(state):
    gpus, saved_env = state
    # Restore the env var *first* – this is what PyTorch reads.
    if saved_env is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = saved_env
    if not gpus:
        return
    try:
        import tensorflow as tf
        tf.config.set_visible_devices(gpus, "GPU")
    except RuntimeError:
        pass


def _move_artifact(src: Path, dst_dir: Path) -> None:
    """Move *src* into *dst_dir*, overwriting if it already exists."""
    target = dst_dir / src.name
    if src.resolve() == target.resolve() or not src.exists():
        return
    if target.exists():
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    shutil.move(str(src), str(target))
    logger.info("Moved %s -> %s", src, target)


def _flatten_saved_model(saved_model_dir: Path, output_dir: Path) -> None:
    """Move all .tflite files from the saved_model dir into output_dir."""
    if not saved_model_dir.is_dir():
        return
    for tflite in saved_model_dir.glob("*.tflite"):
        _move_artifact(tflite, output_dir)


def export(model: str, output_dir: Path) -> Path:
    """Guarantee .pt weights exist inside *output_dir*, downloading if needed.

    Returns the path to the .pt file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    name = model + ".pt"
    target = output_dir / name

    # 1. Already in output/
    if target.exists():
        logger.info("Weights found at %s", target)
        return target

    # 2. Present in the repo root
    nominal = Path(name)
    if nominal.exists() and nominal.resolve() != target.resolve():
        _move_artifact(nominal, output_dir)
        if target.exists():
            return target

    # 3. Download via Ultralytics
    logger.info("Weights not found -- downloading %s via Ultralytics ...", name)
    YOLO(name, task="detect")  # auto-downloads to CWD
    downloaded = Path.cwd() / name
    if downloaded.exists() and downloaded.resolve() != target.resolve():
        _move_artifact(downloaded, output_dir)

    if not target.exists():
        raise FileNotFoundError(f"Could not obtain weights at {target}")
    return target


def export_tflite(
    model: str,
    output_root: Path,
    calibration_yaml: str,
) -> dict[str, str]:
    """Run all needed Ultralytics exports and return a dict of artifact paths.

    Ultralytics generates files inside a ``<stem>_saved_model/`` directory.
    We flatten the .tflite files into *output_root* for convenience.

    Two export calls are made:

    1. **Float export** (``format="tflite"``):
       - ``<stem>_float32.tflite`` — full fp32 precision
       - ``<stem>_float16.tflite`` — fp16 weights, fp32 activations

    2. **Int8 export** (``format="tflite", int8=True, data=...``):
       This single call produces **both** int8 variants via onnx2tf:
       - ``<stem>_int8.tflite`` — **dynamic-range quantization**
         (weights quantized to int8, activations remain fp32 at runtime)
       - ``<stem>_integer_quant.tflite`` — **full integer quantization**
         (weights AND activations quantized to int8 using calibration data)
    """
    pt_path = output_root / f"{model}.pt"
    saved_model_dir = output_root / f"{model}_saved_model"

    model_list = {
        "pt": str(pt_path),
        "fp32": str(output_root / f"{model}_float32.tflite"),
        "fp16": str(output_root / f"{model}_float16.tflite"),
        "int8_dyn": str(output_root / f"{model}_int8.tflite"),
        "int8": str(output_root / f"{model}_integer_quant.tflite"),
    }

    need_float = (
        not Path(model_list["fp32"]).exists()
        or not Path(model_list["fp16"]).exists()
    )
    # A single int8=True call produces BOTH int8_dyn and int8 (full).
    need_int8 = (
        not Path(model_list["int8"]).exists()
        or not Path(model_list["int8_dyn"]).exists()
    )

    if need_float or need_int8:
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            logger.info("PyTorch CUDA context pinned (%s)", torch.cuda.get_device_name(0))

        tf_gpu_state = _hide_tf_gpus()
        try:
            pt_model = YOLO(str(pt_path), task="detect")

            if need_float:
                # Produces: _float32.tflite, _float16.tflite
                logger.info("Exporting TFLite float32 + float16 ...")
                with HeartBeat("PT -> ONNX -> SavedModel -> TFLite (float)"):
                    pt_model.export(format="tflite")
                _flatten_saved_model(saved_model_dir, output_root)

            if need_int8:
                # Produces BOTH variants in one call:
                #   _int8.tflite           = dynamic-range (weights-only int8)
                #   _integer_quant.tflite  = full integer  (weights + activations int8)
                logger.info("Exporting TFLite int8 (dynamic-range + full integer) ...")
                logger.info("  Using calibration data from %s", calibration_yaml)
                with HeartBeat("PT -> ONNX -> SavedModel -> TFLite (int8)"):
                    pt_model.export(format="tflite", int8=True, data=calibration_yaml)
                _flatten_saved_model(saved_model_dir, output_root)
        finally:
            _restore_tf_gpus(tf_gpu_state)
            logger.info(
                "CUDA_VISIBLE_DEVICES after restore: %s  torch.cuda.device_count=%d",
                os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
                torch.cuda.device_count(),
            )
    else:
        logger.info("All TFLite variants already present -- skipping export")

    # onnx2tf downloads a calibration_image_sample_data_*.npy into CWD.
    # Move it into output_root so the repo root stays clean.
    for leftover in Path.cwd().glob("calibration_image_sample_data_*"):
        target = output_root / leftover.name
        if leftover.resolve() != target.resolve() and leftover.exists():
            shutil.move(str(leftover), str(target))
            logger.info("Moved %s -> %s", leftover, target)

    # Log model sizes
    for tag, p in model_list.items():
        sz = file_size_mb(Path(p))
        if sz is not None:
            logger.info("  %s : %.1f MB  (%s)", tag, sz, p)

    return model_list

