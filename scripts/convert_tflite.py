"""
Convert a Keras .h5 checkpoint to TFLite formats (FP32, FP16, optional INT8).
Usage:
    python scripts/convert_tflite.py --h5 palm_ripeness_best_20260311_190150.h5 --rep-data /path/to/train --output-dir models [--labels labels.json] [--preprocess-family mobilenet_v2]

If --labels is omitted, class names are auto-detected from --rep-data (folder names under class subdirs).
"""

import argparse
import datetime
import json
import random
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobilenet_v2
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_input_mobilenet_v3


SUPPORTED_PREPROCESS_FAMILIES = ("mobilenet_v2", "mobilenet_v3", "none")
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _build_mobilenetv3_fallback_model(img_size: int, num_classes: int) -> tf.keras.Model:
    """Rebuild known MobileNetV3 training architecture for H5 weight loading fallback."""
    input_tensor = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    x = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.10),
        ],
        name="data_augmentation",
    )(input_tensor)

    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights=None,
        include_preprocessing=True,
    )
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    output_tensor = tf.keras.layers.Dense(num_classes, activation="softmax", name="ripeness_probs")(x)
    return tf.keras.Model(input_tensor, output_tensor, name="PalmRipeness_MobileNetV3")


def _load_model_resilient(h5_path: str, img_size: int, num_classes: int) -> tf.keras.Model:
    """Load model from H5 with fallback for Keras3 positional-argument deserialization errors."""
    try:
        return tf.keras.models.load_model(h5_path, compile=False)
    except Exception as exc:  # noqa: BLE001
        message = str(exc)

        # QAT checkpoints contain tfmot custom quantization layers. Retry with quantize_scope.
        if "Quantize" in message or "Unknown layer" in message:
            try:
                import tensorflow_model_optimization as tfmot

                with tfmot.quantization.keras.quantize_scope():
                    print("Retrying load with tfmot quantize_scope for QAT model...")
                    return tf.keras.models.load_model(h5_path, compile=False)
            except Exception as qat_exc:  # noqa: BLE001
                print(f"QAT quantize_scope load attempt failed: {qat_exc}")

        if "Only input tensors may be passed as positional arguments" not in message:
            raise

        print("Direct H5 load failed with Keras positional-argument error.")
        print("Falling back to MobileNetV3 architecture reconstruction + weight loading...")
        fallback_model = _build_mobilenetv3_fallback_model(img_size=img_size, num_classes=num_classes)
        fallback_model.load_weights(h5_path)
        return fallback_model


def _apply_preprocess(img: tf.Tensor, preprocess_family: str) -> tf.Tensor:
    family = (preprocess_family or "mobilenet_v2").strip().lower()
    if family == "mobilenet_v2":
        return preprocess_input_mobilenet_v2(img)
    if family == "mobilenet_v3":
        return preprocess_input_mobilenet_v3(img)
    if family == "none":
        if isinstance(img, np.ndarray):
            return img.astype(np.float32)
        return tf.cast(img, tf.float32)
    raise ValueError(
        f"Unsupported preprocess family: {preprocess_family}. "
        f"Expected one of {SUPPORTED_PREPROCESS_FAMILIES}."
    )


def _collect_balanced_representative_paths(
    data_dir: str,
    target_count: int,
    seed: int,
) -> List[Path]:
    root = Path(data_dir)
    class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class folders found under representative data root: {data_dir}")

    per_class = max(1, (target_count + len(class_dirs) - 1) // len(class_dirs))
    rng = random.Random(seed)

    sampled: List[Path] = []
    class_counts = {}
    for class_dir in class_dirs:
        files = sorted(
            [
                p
                for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ]
        )
        if not files:
            class_counts[class_dir.name] = 0
            continue

        shuffled = files[:]
        rng.shuffle(shuffled)
        selected = shuffled[: min(per_class, len(shuffled))]
        sampled.extend(selected)
        class_counts[class_dir.name] = len(selected)

    if not sampled:
        raise ValueError(
            f"No image files found under representative data root: {data_dir}. "
            f"Supported extensions: {SUPPORTED_IMAGE_EXTENSIONS}"
        )

    rng.shuffle(sampled)
    print(
        "[Calibration] strategy=balanced_per_class "
        f"target={target_count} actual={len(sampled)} seed={seed} "
        f"per_class_target={per_class} class_counts={class_counts}"
    )
    return sampled


def make_balanced_representative_dataset(
    data_dir: str,
    img_size: Tuple[int, int],
    take: int = 500,
    preprocess_family: str = "mobilenet_v2",
    seed: int = 42,
) -> Tuple[Callable[[], Iterable[List[np.ndarray]]], int]:
    sampled_paths = _collect_balanced_representative_paths(
        data_dir=data_dir,
        target_count=take,
        seed=seed,
    )

    def representative_dataset() -> Iterable[List[np.ndarray]]:
        for path in sampled_paths:
            img = tf.keras.preprocessing.image.load_img(str(path), target_size=img_size)
            arr = tf.keras.preprocessing.image.img_to_array(img)
            arr = _apply_preprocess(arr, preprocess_family)
            arr = np.expand_dims(arr, axis=0).astype(np.float32)
            yield [arr]

    return representative_dataset, len(sampled_paths)


def detect_labels_from_dir(class_root: str) -> List[str]:
    root = Path(class_root)
    if not root.exists():
        raise FileNotFoundError(f"Class root not found: {class_root}")
    labels = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not labels:
        raise ValueError(f"No class folders found under {class_root}")
    return labels


def save_labels(labels: List[str], output_dir: Path, timestamp: str) -> Path:
    out = output_dir / f"labels_{timestamp}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    return out


def convert_model(
    h5_path: str,
    labels_path: Optional[str],
    output_dir: str,
    rep_data: str,
    img_size: int,
    preprocess_family: str = "mobilenet_v2",
    take_rep: int = 500,
    rep_seed: int = 42,
):
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    preprocess_family = (preprocess_family or "mobilenet_v2").strip().lower()
    if preprocess_family not in SUPPORTED_PREPROCESS_FAMILIES:
        raise ValueError(
            f"Unsupported --preprocess-family '{preprocess_family}'. "
            f"Choose one of: {SUPPORTED_PREPROCESS_FAMILIES}."
        )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_dir_path / f"palm_ripeness_best_{timestamp}"
    if labels_path:
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
    else:
        labels = detect_labels_from_dir(rep_data)
    labels_out = save_labels(labels, output_dir_path, timestamp)

    print(f"Loading model from {h5_path}")
    model = _load_model_resilient(h5_path, img_size=img_size, num_classes=len(labels))

    def _convert(converter: tf.lite.TFLiteConverter, target: str, optimizations=None, rep_fn=None):
        if optimizations:
            converter.optimizations = optimizations
        if rep_fn:
            converter.representative_dataset = rep_fn
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            # Allow float fallback for better compatibility
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32
            # Enable new quantizer for more accurate INT8 weights
            converter.experimental_new_quantizer = True
        tflite_model = converter.convert()
        out_path = f"{base}_{target}.tflite"
        with open(out_path, "wb") as f:
            f.write(tflite_model)
        print(f"Saved {target} model to {out_path}")
        return out_path

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    fp32_path = _convert(converter, "fp32")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_types = [tf.float16]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    fp16_path = _convert(converter, "float16", optimizations=[tf.lite.Optimize.DEFAULT])

    int8_path = None
    rep_actual_count = None
    if rep_data:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        rep_fn, rep_actual_count = make_balanced_representative_dataset(
            rep_data,
            (img_size, img_size),
            take=take_rep,
            preprocess_family=preprocess_family,
            seed=rep_seed,
        )
        int8_path = _convert(converter, "int8", optimizations=[tf.lite.Optimize.DEFAULT], rep_fn=rep_fn)
    else:
        print("Skipping INT8 conversion (no --rep-data provided)")

    manifest = {
        "timestamp": timestamp,
        "source_h5": str(Path(h5_path).resolve()),
        "labels": str(Path(labels_out).resolve()),
        "img_size": img_size,
        "preprocess_family": preprocess_family,
        "class_names": labels,
        "representative_calibration": {
            "strategy": "balanced_per_class",
            "target_count": take_rep if rep_data else None,
            "actual_count": rep_actual_count,
            "seed": rep_seed if rep_data else None,
        },
        "artifacts": {
            "fp32": str(Path(fp32_path).resolve()),
            "float16": str(Path(fp16_path).resolve()),
            "int8": str(Path(int8_path).resolve()) if int8_path else None,
        },
    }
    manifest_path = output_dir_path / f"tflite_manifest_{timestamp}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Labels saved to", labels_out)
    print("Manifest saved to", manifest_path)
    return fp32_path, fp16_path, int8_path, labels_out, manifest_path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Keras checkpoint to TFLite formats")
    parser.add_argument("--h5", required=True, help="Path to .h5 model checkpoint")
    parser.add_argument("--labels", help="Path to labels.json (list of class names). If omitted, derive from --rep-data class folders.")
    parser.add_argument("--output-dir", default="models", help="Directory to store outputs")
    parser.add_argument(
        "--rep-data",
        required=True,
        help="Root folder with class subfolders for INT8 representative dataset and label detection.",
    )
    parser.add_argument("--img-size", type=int, default=224, help="Square image size used during training")
    parser.add_argument(
        "--preprocess-family",
        default="mobilenet_v2",
        choices=SUPPORTED_PREPROCESS_FAMILIES,
        help="Input preprocessing family used during model training.",
    )
    parser.add_argument(
        "--take-rep",
        type=int,
        default=500,
        help="Target representative sample count for balanced INT8 calibration.",
    )
    parser.add_argument(
        "--rep-seed",
        type=int,
        default=42,
        help="Random seed for deterministic balanced representative sampling.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_model(
        args.h5,
        args.labels,
        args.output_dir,
        args.rep_data,
        args.img_size,
        preprocess_family=args.preprocess_family,
        take_rep=args.take_rep,
        rep_seed=args.rep_seed,
    )
