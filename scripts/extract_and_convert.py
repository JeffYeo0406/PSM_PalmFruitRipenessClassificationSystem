"""Convert a legacy H5 checkpoint by rebuilding architecture and loading weights by name.

This is a compatibility fallback for environments where full model deserialization
fails under Keras 3.
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_input_mobilenet_v3


SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fallback H5-to-TFLite converter")
    parser.add_argument("--h5", required=True, help="Path to legacy .h5 checkpoint")
    parser.add_argument("--rep-data", required=True, help="Representative dataset root with class subfolders")
    parser.add_argument("--output-dir", default="models", help="Output directory for TFLite artifacts")
    parser.add_argument("--img-size", type=int, default=224, help="Square input size")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of output classes")
    parser.add_argument("--take-rep", type=int, default=500, help="Target representative samples for balanced INT8 calibration")
    parser.add_argument("--rep-seed", type=int, default=42, help="Random seed for deterministic balanced representative sampling")
    return parser.parse_args()


def detect_labels_from_dir(class_root: Path) -> list[str]:
    labels = sorted([p.name for p in class_root.iterdir() if p.is_dir()])
    if not labels:
        raise ValueError(f"No class folders found under {class_root}")
    return labels


def build_model(img_size: int, num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")

    aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.10),
        ],
        name="data_augmentation",
    )
    x = aug(inputs, training=False)

    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights=None,
        include_preprocessing=True,
    )
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="ripeness_probs")(x)
    model = tf.keras.Model(inputs, outputs, name="PalmRipeness_MobileNetV3")
    return model


def _collect_balanced_representative_paths(data_dir: Path, target_count: int, seed: int) -> list[Path]:
    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class folders found under representative data root: {data_dir}")

    per_class = max(1, (target_count + len(class_dirs) - 1) // len(class_dirs))
    rng = random.Random(seed)

    sampled: list[Path] = []
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
    data_dir: Path,
    img_size: int,
    take: int,
    seed: int,
) -> tuple[Callable[[], Iterable[list[np.ndarray]]], int]:
    sampled_paths = _collect_balanced_representative_paths(
        data_dir=data_dir,
        target_count=take,
        seed=seed,
    )

    def representative_dataset() -> Iterable[list[np.ndarray]]:
        for path in sampled_paths:
            img = tf.keras.preprocessing.image.load_img(str(path), target_size=(img_size, img_size))
            arr = tf.keras.preprocessing.image.img_to_array(img)
            arr = preprocess_input_mobilenet_v3(arr)
            yield [np.expand_dims(arr, axis=0).astype(np.float32)]

    return representative_dataset, len(sampled_paths)


def main() -> int:
    args = parse_args()

    h5_path = Path(args.h5).resolve()
    rep_data = Path(args.rep_data).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not h5_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {h5_path}")
    if not rep_data.exists():
        raise FileNotFoundError(f"Representative data root not found: {rep_data}")

    labels = detect_labels_from_dir(rep_data)
    num_classes = len(labels)

    print("TensorFlow:", tf.__version__)
    print("H5:", h5_path)
    print("Rep data:", rep_data)
    print("Classes:", labels)

    model = build_model(args.img_size, num_classes)

    # Compatibility fallback: load only weights from H5, bypass config deserialization.
    model.load_weights(str(h5_path), by_name=True, skip_mismatch=True)
    _ = model(tf.zeros((1, args.img_size, args.img_size, 3), dtype=tf.float32), training=False)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = out_dir / f"palm_ripeness_best_{ts}"
    labels_path = out_dir / f"labels_{ts}.json"
    manifest_path = out_dir / f"tflite_manifest_{ts}.json"

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32 = converter_fp32.convert()
    fp32_path = Path(f"{base_name}_fp32.tflite")
    fp32_path.write_bytes(tflite_fp32)

    converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_fp16.target_spec.supported_types = [tf.float16]
    tflite_fp16 = converter_fp16.convert()
    fp16_path = Path(f"{base_name}_float16.tflite")
    fp16_path.write_bytes(tflite_fp16)

    converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    rep_fn, rep_actual_count = make_balanced_representative_dataset(
        data_dir=rep_data,
        img_size=args.img_size,
        take=args.take_rep,
        seed=args.rep_seed,
    )
    converter_int8.representative_dataset = rep_fn
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.float32
    converter_int8.inference_output_type = tf.float32
    converter_int8.experimental_new_quantizer = True
    tflite_int8 = converter_int8.convert()
    int8_path = Path(f"{base_name}_int8.tflite")
    int8_path.write_bytes(tflite_int8)

    manifest = {
        "timestamp": ts,
        "source": "weights_fallback_conversion",
        "source_h5": str(h5_path),
        "labels": str(labels_path),
        "img_size": args.img_size,
        "preprocess_family": "mobilenet_v3",
        "class_names": labels,
        "representative_calibration": {
            "strategy": "balanced_per_class",
            "target_count": args.take_rep,
            "actual_count": rep_actual_count,
            "seed": args.rep_seed,
        },
        "artifacts": {
            "fp32": str(fp32_path.resolve()),
            "float16": str(fp16_path.resolve()),
            "int8": str(int8_path.resolve()),
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Saved labels:", labels_path)
    print("Saved manifest:", manifest_path)
    print("Saved FP32:", fp32_path)
    print("Saved FP16:", fp16_path)
    print("Saved INT8:", int8_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
