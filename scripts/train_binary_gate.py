"""
Train and export a binary palm/non-palm detector for stage-1 input gating.

Supported dataset layouts:
1) --data-dir contains train/ and val/:
   data_dir/
     train/<non_palm_class>/*
     train/<palm_class>/*
     val/<non_palm_class>/*
     val/<palm_class>/*

2) --data-dir contains class folders only (automatic split via --val-split):
   data_dir/
     <non_palm_class>/*
     <palm_class>/*
"""

import argparse
import datetime
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE


def _build_datasets(
    data_dir: str,
    img_size: int,
    batch_size: int,
    val_split: float,
    seed: int,
    non_palm_class: str,
    palm_class: str,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    root = Path(data_dir)
    class_names = [non_palm_class, palm_class]
    train_dir = root / "train"
    val_dir = root / "val"

    common = {
        "label_mode": "binary",
        "image_size": (img_size, img_size),
        "batch_size": batch_size,
        "class_names": class_names,
        "seed": seed,
    }

    if train_dir.exists() and val_dir.exists():
        train_ds = tf.keras.utils.image_dataset_from_directory(str(train_dir), shuffle=True, **common)
        val_ds = tf.keras.utils.image_dataset_from_directory(str(val_dir), shuffle=False, **common)
    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            str(root),
            validation_split=val_split,
            subset="training",
            shuffle=True,
            **common,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            str(root),
            validation_split=val_split,
            subset="validation",
            shuffle=False,
            **common,
        )

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names


def _build_model(img_size: int, dropout: float) -> tf.keras.Model:
    data_aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
        ],
        name="binary_gate_aug",
    )

    backbone = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="input_rgb")
    x = data_aug(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="palm_probability")(x)
    model = tf.keras.Model(inputs, outputs, name="palm_binary_gate")
    return model


def _compile(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def _collect_validation_predictions(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
) -> Tuple[np.ndarray, np.ndarray]:
    y_true: List[np.ndarray] = []
    y_prob: List[np.ndarray] = []
    for images, labels in val_ds:
        probs = model.predict(images, verbose=0).reshape(-1)
        y_prob.append(probs)
        y_true.append(labels.numpy().reshape(-1))
    return np.concatenate(y_true).astype(np.int32), np.concatenate(y_prob).astype(np.float32)


def _metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int32)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(len(y_true), 1)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    best = None
    for t in np.linspace(0.30, 0.90, 61):
        current = _metrics_at_threshold(y_true, y_prob, float(t))
        if best is None:
            best = current
            continue
        better = (
            current["f1"] > best["f1"]
            or (current["f1"] == best["f1"] and current["recall"] > best["recall"])
            or (
                current["f1"] == best["f1"]
                and current["recall"] == best["recall"]
                and abs(current["threshold"] - 0.60) < abs(best["threshold"] - 0.60)
            )
        )
        if better:
            best = current
    return best if best is not None else _metrics_at_threshold(y_true, y_prob, 0.60)


def _representative_dataset(train_ds: tf.data.Dataset, max_samples: int) -> Iterable[List[tf.Tensor]]:
    emitted = 0
    for images, _ in train_ds:
        batch_size = int(images.shape[0])
        for i in range(batch_size):
            yield [tf.expand_dims(images[i], axis=0)]
            emitted += 1
            if emitted >= max_samples:
                return


def _convert_tflite(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    out_base: Path,
    rep_samples: int,
) -> Tuple[Path, Path]:
    fp32_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    fp32_model = fp32_converter.convert()
    fp32_path = out_base.with_name(out_base.name + "_fp32.tflite")
    fp32_path.write_bytes(fp32_model)

    int8_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    int8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    int8_converter.representative_dataset = lambda: _representative_dataset(train_ds, rep_samples)
    int8_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    int8_converter.inference_input_type = tf.float32
    int8_converter.inference_output_type = tf.float32
    int8_converter.experimental_new_quantizer = True
    int8_model = int8_converter.convert()
    int8_path = out_base.with_name(out_base.name + "_int8.tflite")
    int8_path.write_bytes(int8_model)

    return fp32_path, int8_path


def train_binary_gate(args: argparse.Namespace) -> Dict[str, str]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, class_names = _build_datasets(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        non_palm_class=args.non_palm_class,
        palm_class=args.palm_class,
    )

    model = _build_model(img_size=args.img_size, dropout=args.dropout)
    _compile(model, learning_rate=args.lr_warmup)

    print("Warm-up training...")
    model.fit(train_ds, validation_data=val_ds, epochs=args.warmup_epochs, verbose=1)

    if args.finetune_epochs > 0:
        print("Fine-tuning...")
        backbone = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) and layer.name.startswith("mobilenetv2"):
                backbone = layer
                break
        if backbone is not None:
            backbone.trainable = True
            if args.unfreeze_layers > 0:
                for layer in backbone.layers[:-args.unfreeze_layers]:
                    layer.trainable = False
        _compile(model, learning_rate=args.lr_finetune)
        model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs, verbose=1)

    y_true, y_prob = _collect_validation_predictions(model, val_ds)
    metrics_default = _metrics_at_threshold(y_true, y_prob, args.default_threshold)
    metrics_recommended = _best_threshold(y_true, y_prob)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = output_dir / f"palm_presence_binary_{timestamp}"
    h5_path = out_base.with_suffix(".h5")
    model.save(h5_path, include_optimizer=False)

    fp32_path, int8_path = _convert_tflite(model, train_ds, out_base, rep_samples=args.rep_samples)

    canonical_model = output_dir / "palm_presence_binary.tflite"
    shutil.copy2(int8_path, canonical_model)

    manifest = {
        "timestamp": timestamp,
        "data_dir": str(Path(args.data_dir).resolve()),
        "class_names": class_names,
        "palm_class": args.palm_class,
        "non_palm_class": args.non_palm_class,
        "palm_class_index": 1,
        "default_threshold": float(args.default_threshold),
        "recommended_threshold": float(metrics_recommended["threshold"]),
        "validation_metrics": {
            "default": metrics_default,
            "recommended": metrics_recommended,
        },
        "artifacts": {
            "h5": str(h5_path.resolve()),
            "fp32": str(fp32_path.resolve()),
            "int8": str(int8_path.resolve()),
            "canonical_tflite": str(canonical_model.resolve()),
        },
    }

    manifest_path = output_dir / f"palm_presence_binary_manifest_{timestamp}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Saved binary gate artifacts:")
    print("  H5:", h5_path)
    print("  FP32:", fp32_path)
    print("  INT8:", int8_path)
    print("  Canonical:", canonical_model)
    print("  Manifest:", manifest_path)
    print("Recommended threshold:", f"{metrics_recommended['threshold']:.3f}")

    return {
        "h5": str(h5_path),
        "fp32": str(fp32_path),
        "int8": str(int8_path),
        "canonical": str(canonical_model),
        "manifest": str(manifest_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export palm/non-palm binary gate model")
    parser.add_argument("--data-dir", required=True, help="Dataset root for binary gate training")
    parser.add_argument("--output-dir", default="models", help="Output directory for exported artifacts")
    parser.add_argument("--img-size", type=int, default=224, help="Square image size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split when no val/ folder exists")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warm-up epochs with frozen backbone")
    parser.add_argument("--finetune-epochs", type=int, default=3, help="Fine-tune epochs")
    parser.add_argument("--unfreeze-layers", type=int, default=20, help="Number of backbone layers to unfreeze")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout ratio")
    parser.add_argument("--lr-warmup", type=float, default=1e-4, help="Warm-up learning rate")
    parser.add_argument("--lr-finetune", type=float, default=5e-6, help="Fine-tune learning rate")
    parser.add_argument("--rep-samples", type=int, default=300, help="Representative samples for INT8 conversion")
    parser.add_argument("--default-threshold", type=float, default=0.60, help="Default palm gate threshold")
    parser.add_argument("--non-palm-class", default="non_palm", help="Negative class folder name")
    parser.add_argument("--palm-class", default="palm", help="Positive class folder name")
    return parser.parse_args()


if __name__ == "__main__":
    train_binary_gate(parse_args())
