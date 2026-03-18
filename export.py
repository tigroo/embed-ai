"""
YOLO export: download weights and produce TFLite + ONNX variants.
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

    2. **Int8 export** (``format="tflite", int8=True, data="coco128.yaml"``):
       Uses COCO-128 (auto-downloaded by Ultralytics) as calibration —
       128 diverse images give much better activation ranges than our few
       domain videos.  This single call produces three int8 variants via
       onnx2tf:
       - ``<stem>_int8.tflite`` — **dynamic-range quantization**
         (weights quantized to int8, activations remain fp32 at runtime)
       - ``<stem>_integer_quant.tflite`` — **integer quantization**
         (weights + activations int8 where supported, float fallback otherwise)
       - ``<stem>_full_integer_quant.tflite`` — **full integer quantization**
         (everything forced to int8, no fallback — most aggressive)
    """
    pt_path = output_root / f"{model}.pt"
    saved_model_dir = output_root / f"{model}_saved_model"

    model_list = {
        "pt":        str(pt_path),
        "fp32":      str(output_root / f"{model}_float32.tflite"),
        "fp16":      str(output_root / f"{model}_float16.tflite"),
        "int8_dyn":  str(output_root / f"{model}_int8.tflite"),
        "int8_full": str(output_root / f"{model}_full_integer_quant.tflite"),
    }

    need_float = (
        not Path(model_list["fp32"]).exists()
        or not Path(model_list["fp16"]).exists()
    )
    # A single int8=True call produces int8_dyn + int8_full via onnx2tf.
    # (also generates _integer_quant.tflite but we ignore it — same
    # breakage as int8_full, just with float I/O.)
    need_int8 = (
        not Path(model_list["int8_dyn"]).exists()
        or not Path(model_list["int8_full"]).exists()
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
                # Uses coco128 (128 diverse COCO images) for calibration.
                # Produces all three int8 variants in one call:
                #   _int8.tflite                = dynamic-range (weights-only int8)
                #   _integer_quant.tflite       = integer quant (int8 + float fallback)
                #   _full_integer_quant.tflite  = full integer  (everything int8)
                _INT8_CAL_DATA = "coco128.yaml"
                logger.info("Exporting TFLite int8 (dynamic + integer + full) ...")
                logger.info(
                    "  Calibration: %s (auto-downloaded by Ultralytics)",
                    _INT8_CAL_DATA,
                )
                with HeartBeat("PT -> ONNX -> SavedModel -> TFLite (int8)"):
                    pt_model.export(format="tflite", int8=True, data=_INT8_CAL_DATA)
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

    # onnx2tf downloads calibration_image_sample_data_*.npy into CWD and
    # sometimes into the saved_model directory.  Sweep both locations so
    # the repo root stays clean.
    for search_dir in [Path.cwd(), saved_model_dir]:
        for leftover in search_dir.glob("calibration_image_sample_data_*"):
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


def _get_onnx_opset(onnx_path: Path) -> int:
    """Return the ai.onnx opset version of an ONNX model."""
    import onnx as _onnx
    m = _onnx.load(str(onnx_path))
    return max(
        (o.version for o in m.opset_import if o.domain in ("", "ai.onnx")),
        default=0,
    )


# Highest opset fully supported by ONNX Runtime Web's WebGL + WASM backends.
# Split@18+ and several other ops are NOT implemented in the browser runtime.
_WEB_ONNX_OPSET = 17


def export_onnx_for_pwa(output_root: Path, model: str) -> tuple[Path, Path]:
    """Create web-compatible ONNX models directly in ``docs/pwa/models/``.

    Ultralytics exports the ONNX intermediate at opset 20, but ONNX
    Runtime Web (WebGL / WASM) doesn't support ``Split`` at that opset.
    This function exports a fresh ONNX at opset ≤ 17, moves it straight
    into the PWA directory, and quantizes the INT8 variant in-place.

    Returns ``(pwa_fp32_path, pwa_int8_path)``.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    pt_path = output_root / f"{model}.pt"
    pwa_dir = Path("docs") / "pwa" / "models"
    pwa_dir.mkdir(parents=True, exist_ok=True)

    pwa_fp32 = pwa_dir / f"{model}_fp32.onnx"
    pwa_int8 = pwa_dir / f"{model}_int8.onnx"

    # ── Step 1: FP32 ONNX at web-compatible opset ────────────────────
    need_fp32 = True
    if pwa_fp32.exists():
        opset = _get_onnx_opset(pwa_fp32)
        if opset <= _WEB_ONNX_OPSET:
            need_fp32 = False
            logger.info(
                "PWA FP32 already at opset %d: %s (%.1f MB)",
                opset, pwa_fp32, file_size_mb(pwa_fp32),
            )
        else:
            logger.info(
                "PWA ONNX opset %d > %d -- re-exporting with opset %d",
                opset, _WEB_ONNX_OPSET, _WEB_ONNX_OPSET,
            )
            pwa_fp32.unlink()
            if pwa_int8.exists():
                pwa_int8.unlink()  # must re-quantize after opset change

    if need_fp32:
        if not pt_path.exists():
            raise FileNotFoundError(f"PT weights not found at {pt_path}")
        logger.info("Exporting ONNX (opset %d) for PWA ...", _WEB_ONNX_OPSET)
        yolo_model = YOLO(str(pt_path), task="detect")
        yolo_model.export(format="onnx", opset=_WEB_ONNX_OPSET)
        # Ultralytics writes next to the .pt → output/yolo26n.onnx
        generated = pt_path.with_suffix(".onnx")
        if generated.exists():
            shutil.move(str(generated), str(pwa_fp32))
            logger.info("Moved %s -> %s", generated, pwa_fp32)
        logger.info(
            "  PWA FP32 : %.1f MB  opset %d  (%s)",
            file_size_mb(pwa_fp32), _WEB_ONNX_OPSET, pwa_fp32,
        )

    # ── Step 2: INT8 (backbone quantized, head stays FP32) ───────────
    if pwa_int8.exists():
        logger.info(
            "PWA INT8 already present: %s (%.1f MB)",
            pwa_int8, file_size_mb(pwa_int8),
        )
        return pwa_fp32, pwa_int8

    head_nodes = _find_head_nodes(pwa_fp32)
    logger.info(
        "Quantizing ONNX: backbone INT8 (dynamic), "
        "head FP32 (%d nodes excluded) ...",
        len(head_nodes),
    )
    quantize_dynamic(
        model_input=str(pwa_fp32),
        model_output=str(pwa_int8),
        weight_type=QuantType.QUInt8,
        nodes_to_exclude=head_nodes,
    )
    logger.info("  PWA FP32 : %.1f MB  (%s)", file_size_mb(pwa_fp32), pwa_fp32)
    logger.info("  PWA INT8 : %.1f MB  (%s)", file_size_mb(pwa_int8), pwa_int8)

    return pwa_fp32, pwa_int8


def _find_head_nodes(onnx_path: Path) -> list[str]:
    """Return node names belonging to the YOLO detection head.

    In YOLO v26 the detection head is ``/model.23/`` — it contains the
    one2one_cv2 (boxes), one2one_cv3 (classes) convolutions and the
    post-processing ops (Sigmoid, TopK, NMS-like gather/sort).
    Keeping these in fp32 preserves confidence precision.
    """
    import onnx
    graph = onnx.load(str(onnx_path)).graph
    head = [n.name for n in graph.node if n.name.startswith("/model.23/")]
    if head:
        logger.info("Detection head: %d nodes under /model.23/ kept in FP32", len(head))
    return head

