"""Quantization-aware fine-tuning workflow for MobileNetV3.

This script loads a selected MobileNetV3 checkpoint via weight-loading fallback,
performs QAT fine-tuning, and saves best/final QAT checkpoints for TFLite conversion.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import tf_keras as keras
from sklearn.metrics import precision_recall_fscore_support

try:
    import tensorflow_model_optimization as tfmot
except Exception as exc:  # noqa: BLE001
    raise RuntimeError(
        "tensorflow-model-optimization is required for QAT. "
        "Install with: pip install tensorflow-model-optimization"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = Path(r"C:\Users\jeffy\Documents\PSM\Dataset1")
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QAT fine-tuning for MobileNetV3 checkpoint")
    parser.add_argument(
        "--h5",
        required=True,
        help="Path to source checkpoint (.h5 legacy model/weights file)",
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="Dataset root containing Train and Test")
    parser.add_argument("--train-root", default="", help="Optional explicit Train folder")
    parser.add_argument("--test-root", default="", help="Optional explicit Test folder")
    parser.add_argument("--output-dir", default="saved_models", help="Directory for QAT checkpoints")
    parser.add_argument("--reports-dir", default="reports", help="Directory for QAT summary output")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8, help="QAT fine-tuning epochs")
    parser.add_argument("--qat-lr", type=float, default=1e-5, help="Learning rate during QAT")
    parser.add_argument(
        "--unfreeze-layers",
        type=int,
        default=30,
        help="Number of top MobileNetV3 layers to unfreeze before QAT",
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable augmentation layers in QAT model for compatibility/stability",
    )
    parser.add_argument(
        "--export-tflite",
        action="store_true",
        help="Also export fp32/fp16/int8 TFLite artifacts from the trained QAT model",
    )
    parser.add_argument(
        "--tflite-output-dir",
        default="models",
        help="Directory to write exported TFLite artifacts when --export-tflite is set",
    )
    parser.add_argument(
        "--rep-data",
        default="",
        help="Representative dataset root for INT8 export (defaults to train root)",
    )
    parser.add_argument("--take-rep", type=int, default=500, help="Representative samples target for INT8 export")
    parser.add_argument("--rep-seed", type=int, default=42, help="Random seed for representative sampling")
    return parser.parse_args()


def as_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def list_class_names(root: Path) -> list[str]:
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def count_images(root: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def validate_dataset_layout(train_root: Path, test_root: Path) -> tuple[list[str], int, int]:
    if not train_root.exists():
        raise FileNotFoundError(f"Train directory not found: {train_root}")
    if not test_root.exists():
        raise FileNotFoundError(f"Test directory not found: {test_root}")

    class_names = list_class_names(train_root)
    test_class_names = list_class_names(test_root)
    if not class_names:
        raise ValueError(f"No class folders found under train root: {train_root}")
    if class_names != test_class_names:
        raise ValueError(f"Train/Test class folders mismatch: {class_names} vs {test_class_names}")

    return class_names, count_images(train_root), count_images(test_root)


def make_datasets(
    train_root: Path,
    test_root: Path,
    class_names: list[str],
    img_size: tuple[int, int],
    batch_size: int,
    val_split: float,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    autotune = tf.data.AUTOTUNE

    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        train_root,
        labels="inferred",
        label_mode="categorical",
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        train_root,
        labels="inferred",
        label_mode="categorical",
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
    )
    test_ds_raw = tf.keras.utils.image_dataset_from_directory(
        test_root,
        labels="inferred",
        label_mode="categorical",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )

    if train_ds_raw.class_names != class_names:
        raise ValueError(f"Class order mismatch. expected={class_names}, got={train_ds_raw.class_names}")

    def preprocess_xy(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        # Keep raw image scale for MobileNetV3 include_preprocessing=True.
        return tf.cast(x, tf.float32), y

    train_ds = train_ds_raw.map(preprocess_xy, num_parallel_calls=autotune).ignore_errors().prefetch(autotune)
    val_ds = val_ds_raw.map(preprocess_xy, num_parallel_calls=autotune).ignore_errors().prefetch(autotune)
    test_ds = test_ds_raw.map(preprocess_xy, num_parallel_calls=autotune).ignore_errors().prefetch(autotune)
    return train_ds, val_ds, test_ds


def build_mobilenetv3_model(
    num_classes: int,
    img_size: tuple[int, int],
    include_augmentation: bool,
) -> tuple[keras.Model, keras.Model]:
    # QAT compatibility: use a non-nested Functional graph (base.input -> base.output)
    # because quantizing a nested Model inside another Model is unsupported.
    if include_augmentation:
        print("QAT note: augmentation is disabled for quantization compatibility.")

    base_model = keras.applications.MobileNetV3Small(
        input_shape=(*img_size, 3),
        include_top=False,
        weights=None,
        include_preprocessing=True,
    )
    base_model.trainable = False

    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.30)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.20)(x)
    output_tensor = keras.layers.Dense(num_classes, activation="softmax", name="ripeness_probs")(x)
    model = keras.Model(base_model.input, output_tensor, name="PalmRipeness_MobileNetV3")
    return model, base_model


def build_source_checkpoint_model(
    num_classes: int,
    img_size: tuple[int, int],
) -> tuple[keras.Model, keras.Model]:
    input_tensor = keras.Input(shape=(*img_size, 3), name="image")
    x = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.15),
            keras.layers.RandomZoom(0.10),
        ],
        name="data_augmentation",
    )(input_tensor)

    base_model = keras.applications.MobileNetV3Small(
        input_shape=(*img_size, 3),
        include_top=False,
        weights=None,
        include_preprocessing=True,
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.30)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.20)(x)
    output_tensor = keras.layers.Dense(num_classes, activation="softmax", name="ripeness_probs")(x)
    model = keras.Model(input_tensor, output_tensor, name="PalmRipeness_MobileNetV3")
    return model, base_model


def _copy_matching_weights(src_layer: keras.layers.Layer, dst_layer: keras.layers.Layer) -> bool:
    src_weights = src_layer.get_weights()
    dst_weights = dst_layer.get_weights()

    if not src_weights or not dst_weights:
        return False
    if len(src_weights) != len(dst_weights):
        return False
    if any(sw.shape != dw.shape for sw, dw in zip(src_weights, dst_weights)):
        return False

    dst_layer.set_weights(src_weights)
    return True


def transfer_weights_from_source(
    source_model: keras.Model,
    source_base: keras.Model,
    target_model: keras.Model,
    target_base: keras.Model,
) -> tuple[int, int]:
    copied = 0
    total = 0

    source_base_layers = {layer.name: layer for layer in source_base.layers}
    for dst_layer in target_base.layers:
        if not dst_layer.get_weights():
            continue
        total += 1
        src_layer = source_base_layers.get(dst_layer.name)
        if src_layer is not None and _copy_matching_weights(src_layer, dst_layer):
            copied += 1

    source_head_layers = [
        layer
        for layer in source_model.layers
        if layer is not source_base and layer.get_weights()
    ]
    target_base_layer_names = {layer.name for layer in target_base.layers}
    target_head_layers = [
        layer
        for layer in target_model.layers
        if layer.name not in target_base_layer_names and layer.get_weights()
    ]

    used_source_idx: set[int] = set()
    for dst_layer in target_head_layers:
        total += 1
        matched = False
        for idx, src_layer in enumerate(source_head_layers):
            if idx in used_source_idx:
                continue
            if src_layer.__class__ is not dst_layer.__class__:
                continue
            if _copy_matching_weights(src_layer, dst_layer):
                used_source_idx.add(idx)
                copied += 1
                matched = True
                break
        if not matched:
                continue

    return copied, total

def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def configure_fine_tune(base_model: keras.Model, unfreeze_layers: int) -> None:
    base_model.trainable = True
    freeze_until = max(0, len(base_model.layers) - unfreeze_layers)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    for layer in base_model.layers[freeze_until:]:
        layer.trainable = True


def evaluate_model(model: keras.Model, test_ds: tf.data.Dataset) -> dict[str, Any]:
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)

    y_true: list[int] = []
    y_pred: list[int] = []
    for x_batch, y_batch in test_ds:
        probs = model.predict(x_batch, verbose=0)
        y_true.extend(np.argmax(y_batch.numpy(), axis=1).tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "test_loss": float(test_loss),
        "accuracy": float(test_acc),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
    }


def _collect_balanced_representative_paths(data_dir: Path, target_count: int, seed: int) -> list[Path]:
    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class folders found under representative data root: {data_dir}")

    per_class = max(1, (target_count + len(class_dirs) - 1) // len(class_dirs))
    rng = random.Random(seed)

    sampled: list[Path] = []
    class_counts: dict[str, int] = {}
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
):
    sampled_paths = _collect_balanced_representative_paths(data_dir=data_dir, target_count=take, seed=seed)

    def representative_data_gen():
        for image_path in sampled_paths:
            img = tf.io.read_file(str(image_path))
            img = tf.io.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, [img_size, img_size])
            img = tf.cast(img, tf.float32)
            yield [tf.expand_dims(img, axis=0)]

    return representative_data_gen, len(sampled_paths)


def export_tflite_artifacts(
    model: keras.Model,
    class_names: list[str],
    output_dir: Path,
    rep_data_dir: Path,
    img_size: int,
    take_rep: int,
    rep_seed: int,
    timestamp: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_path = output_dir / f"labels_qat_{timestamp}.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    fp32_path = output_dir / f"palm_ripeness_qat_{timestamp}_fp32.tflite"
    fp16_path = output_dir / f"palm_ripeness_qat_{timestamp}_float16.tflite"
    int8_path = output_dir / f"palm_ripeness_qat_{timestamp}_int8.tflite"

    print("Converting QAT model to FP32 TFLite...")
    converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32 = converter_fp32.convert()
    fp32_path.write_bytes(tflite_fp32)

    print("Converting QAT model to FP16 TFLite...")
    converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_fp16.target_spec.supported_types = [tf.float16]
    tflite_fp16 = converter_fp16.convert()
    fp16_path.write_bytes(tflite_fp16)

    print("Converting QAT model to INT8 TFLite...")
    converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.experimental_new_quantizer = True
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.float32
    converter_int8.inference_output_type = tf.float32
    tflite_int8 = converter_int8.convert()
    int8_path.write_bytes(tflite_int8)

    manifest = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "labels": str(labels_path.resolve()),
        "fp32": str(fp32_path.resolve()),
        "fp16": str(fp16_path.resolve()),
        "int8": str(int8_path.resolve()),
        "representative_calibration": {
            "strategy": "qat_no_representative_required",
            "requested_take": int(take_rep),
            "actual_samples": 0,
            "rep_seed": int(rep_seed),
            "rep_data": str(rep_data_dir.resolve()),
        },
    }

    manifest_path = output_dir / f"qat_tflite_manifest_{timestamp}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return {
        "labels": str(labels_path.resolve()),
        "fp32": str(fp32_path.resolve()),
        "fp16": str(fp16_path.resolve()),
        "int8": str(int8_path.resolve()),
        "manifest": str(manifest_path.resolve()),
    }


def main() -> int:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    dataset_root = as_project_path(args.dataset_root)
    train_root = as_project_path(args.train_root) if args.train_root else dataset_root / "Train"
    test_root = as_project_path(args.test_root) if args.test_root else dataset_root / "Test"
    h5_path = as_project_path(args.h5)
    output_dir = as_project_path(args.output_dir)
    reports_dir = as_project_path(args.reports_dir)
    tflite_output_dir = as_project_path(args.tflite_output_dir)
    rep_data_dir = as_project_path(args.rep_data) if args.rep_data else train_root

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    class_names, train_image_count, test_image_count = validate_dataset_layout(train_root, test_root)
    img_size = (args.img_size, args.img_size)

    print("=== QAT MobileNetV3 Fine-tuning ===")
    print("Source checkpoint:", h5_path)
    print("Train root:", train_root)
    print("Test root:", test_root)
    print("Class names:", class_names)
    print("Train images:", train_image_count)
    print("Test images:", test_image_count)

    train_ds, val_ds, test_ds = make_datasets(
        train_root=train_root,
        test_root=test_root,
        class_names=class_names,
        img_size=img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
    )

    model, base_model = build_mobilenetv3_model(
        num_classes=len(class_names),
        img_size=img_size,
        include_augmentation=not args.no_augmentation,
    )

    if not h5_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {h5_path}")

    # Load through original source architecture, then transfer into non-nested QAT model.
    source_model, source_base = build_source_checkpoint_model(
        num_classes=len(class_names),
        img_size=img_size,
    )
    source_model.load_weights(str(h5_path))
    copied_layers, total_layers = transfer_weights_from_source(
        source_model=source_model,
        source_base=source_base,
        target_model=model,
        target_base=base_model,
    )
    print(f"Transferred weights: {copied_layers}/{total_layers} weighted layers")
    if copied_layers == 0:
        raise RuntimeError("Failed to transfer any checkpoint weights into QAT model.")

    _ = model(tf.zeros((1, args.img_size, args.img_size, 3), dtype=tf.float32), training=False)

    configure_fine_tune(base_model, unfreeze_layers=args.unfreeze_layers)

    compile_model(model, learning_rate=args.qat_lr)
    pre_qat_metrics = evaluate_model(model, test_ds)
    print("Pre-QAT accuracy:", round(pre_qat_metrics["accuracy"], 4))
    if pre_qat_metrics["accuracy"] < 0.6:
        raise RuntimeError(
            "Pre-QAT accuracy is unexpectedly low; aborting to avoid invalid QAT run. "
            f"Observed={pre_qat_metrics['accuracy']:.4f}"
        )

    # MobileNetV3 includes TFOpLambda hard-swish blocks that are unstable under full-model
    # tfmot quantize_model in this environment. Use stable head-only QAT annotation.
    def apply_quantization_to_layer(layer):
        if isinstance(layer, keras.layers.Dense):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        return layer

    annotated_model = keras.models.clone_model(model, clone_function=apply_quantization_to_layer)
    annotated_model.set_weights(model.get_weights())
    qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    compile_model(qat_model, learning_rate=args.qat_lr)
    qat_init_metrics = evaluate_model(qat_model, test_ds)
    print("QAT-init accuracy:", round(qat_init_metrics["accuracy"], 4))
    if qat_init_metrics["accuracy"] < 0.6:
        raise RuntimeError(
            "QAT initialization accuracy is unexpectedly low; aborting. "
            f"Observed={qat_init_metrics['accuracy']:.4f}"
        )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_qat_path = output_dir / f"palm_ripeness_qat_best_{timestamp}.keras"
    final_qat_path = output_dir / f"palm_ripeness_qat_final_{timestamp}.keras"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_qat_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = qat_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    qat_model.save(str(final_qat_path))
    post_qat_metrics = evaluate_model(qat_model, test_ds)

    tflite_artifacts: dict[str, str] | None = None
    if args.export_tflite:
        if not rep_data_dir.exists():
            raise FileNotFoundError(f"Representative data directory not found: {rep_data_dir}")
        tflite_artifacts = export_tflite_artifacts(
            model=qat_model,
            class_names=class_names,
            output_dir=tflite_output_dir,
            rep_data_dir=rep_data_dir,
            img_size=args.img_size,
            take_rep=args.take_rep,
            rep_seed=args.rep_seed,
            timestamp=timestamp,
        )

    summary = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_checkpoint": str(h5_path.resolve()),
        "dataset": {
            "train_root": str(train_root.resolve()),
            "test_root": str(test_root.resolve()),
            "class_names": class_names,
            "train_images": train_image_count,
            "test_images": test_image_count,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "val_split": args.val_split,
        },
        "qat_config": {
            "epochs": args.epochs,
            "qat_lr": args.qat_lr,
            "unfreeze_layers": args.unfreeze_layers,
            "seed": args.seed,
            "augmentation_enabled": not args.no_augmentation,
        },
        "artifacts": {
            "best_qat_model": str(best_qat_path.resolve()),
            "final_qat_model": str(final_qat_path.resolve()),
            "tflite": tflite_artifacts,
        },
        "metrics": {
            "pre_qat": pre_qat_metrics,
            "post_qat": post_qat_metrics,
        },
        "training": {
            "epochs_completed": len(history.history.get("loss", [])),
            "best_val_loss": float(np.min(history.history.get("val_loss", [np.nan]))),
        },
    }

    summary_path = reports_dir / f"qat_mobilenetv3_summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Best QAT model:", best_qat_path)
    print("Final QAT model:", final_qat_path)
    if tflite_artifacts:
        print("QAT TFLite manifest:", tflite_artifacts["manifest"])
    print("QAT summary:", summary_path)
    print("Post-QAT accuracy:", round(post_qat_metrics["accuracy"], 4))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())





