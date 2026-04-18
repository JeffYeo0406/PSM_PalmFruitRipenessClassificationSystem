"""
Convert a Keras .h5 checkpoint to TFLite formats (FP32, FP16, optional INT8).
Usage:
    python scripts/convert_tflite.py --h5 palm_ripeness_best_20260311_190150.h5 --rep-data /path/to/train --output-dir models [--labels labels.json] [--preprocess-family mobilenet_v2]

If --labels is omitted, class names are auto-detected from --rep-data (folder names under class subdirs).
"""

import argparse
import datetime
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobilenet_v2
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_input_mobilenet_v3


SUPPORTED_PREPROCESS_FAMILIES = ("mobilenet_v2", "mobilenet_v3", "none")


def _apply_preprocess(img: tf.Tensor, preprocess_family: str) -> tf.Tensor:
    family = (preprocess_family or "mobilenet_v2").strip().lower()
    if family == "mobilenet_v2":
        return preprocess_input_mobilenet_v2(img)
    if family == "mobilenet_v3":
        return preprocess_input_mobilenet_v3(img)
    if family == "none":
        return tf.cast(img, tf.float32)
    raise ValueError(
        f"Unsupported preprocess family: {preprocess_family}. "
        f"Expected one of {SUPPORTED_PREPROCESS_FAMILIES}."
    )


def representative_dataset(
    data_dir: str,
    img_size: Tuple[int, int],
    take: int = 500,
    preprocess_family: str = "mobilenet_v2",
) -> Iterable:
    files = tf.data.Dataset.list_files(str(Path(data_dir) / "*" / "*.*"), shuffle=False)

    def _load(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, img_size)
        img = _apply_preprocess(img, preprocess_family)
        return tf.expand_dims(img, 0)

    for batch in files.map(_load).take(take):
        yield [batch]


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
    model = tf.keras.models.load_model(h5_path, compile=False)

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
    if rep_data:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        rep_fn = lambda: representative_dataset(
            rep_data,
            (img_size, img_size),
            preprocess_family=preprocess_family,
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
    )
