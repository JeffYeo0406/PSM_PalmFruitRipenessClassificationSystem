"""Run 7-flow EfficientNetB0 reproduction against historical EfficientNetB0 experiment flow.

This script executes the same top-to-bottom configuration sequence recorded in
reports/experiment_log.csv, writes results to a separate CSV, and selects the
best-accuracy EfficientNetB0 run for downstream quantization.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as preprocess_input_efficientnet


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = Path(r"C:\Users\jeffy\Documents\PSM\Dataset1")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG_COLUMNS = [
    "timestamp",
    "run_mode",
    "epochs_config",
    "fine_tune_epochs",
    "batch_size",
    "val_split",
    "initial_learning_rate",
    "final_learning_rate",
    "backbone_unfrozen",
    "train_images",
    "test_images",
    "num_classes",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "best_model_file",
    "excel_report",
    "notes",
]


@dataclass(frozen=True)
class RunProfile:
    name: str
    run_mode: str
    epochs_config: int
    fine_tune_epochs: int
    initial_lr: float
    final_lr: float
    smoke: bool
    backbone_unfrozen: bool


PROFILES = [
    RunProfile(
        name="01_smoke_test",
        run_mode="smoke_test",
        epochs_config=10,
        fine_tune_epochs=0,
        initial_lr=1e-4,
        final_lr=1e-4,
        smoke=True,
        backbone_unfrozen=False,
    ),
    RunProfile(
        name="02_full_train_e10",
        run_mode="full_train",
        epochs_config=10,
        fine_tune_epochs=0,
        initial_lr=1e-4,
        final_lr=1e-4,
        smoke=False,
        backbone_unfrozen=False,
    ),
    RunProfile(
        name="03_full_train_e10_ft5",
        run_mode="full_train + fine_tune",
        epochs_config=10,
        fine_tune_epochs=5,
        initial_lr=5e-6,
        final_lr=1e-4,
        smoke=False,
        backbone_unfrozen=True,
    ),
    RunProfile(
        name="04_full_train_e10_ft15",
        run_mode="full_train + fine_tune",
        epochs_config=10,
        fine_tune_epochs=15,
        initial_lr=5e-6,
        final_lr=1e-4,
        smoke=False,
        backbone_unfrozen=True,
    ),
    RunProfile(
        name="05_full_train_e30",
        run_mode="full_train",
        epochs_config=30,
        fine_tune_epochs=0,
        initial_lr=1e-4,
        final_lr=1e-4,
        smoke=False,
        backbone_unfrozen=False,
    ),
    RunProfile(
        name="06_full_train_e30_ft5",
        run_mode="full_train + fine_tune",
        epochs_config=30,
        fine_tune_epochs=5,
        initial_lr=5e-6,
        final_lr=5e-5,
        smoke=False,
        backbone_unfrozen=True,
    ),
    RunProfile(
        name="07_full_train_e30_ft15",
        run_mode="full_train + fine_tune",
        epochs_config=30,
        fine_tune_epochs=15,
        initial_lr=5e-6,
        final_lr=5e-5,
        smoke=False,
        backbone_unfrozen=True,
    ),
]


def select_profiles(all_profiles: list[RunProfile], requested: str) -> list[RunProfile]:
    if not requested.strip():
        return all_profiles

    name_to_profile = {profile.name: profile for profile in all_profiles}
    requested_names = [name.strip() for name in requested.split(",") if name.strip()]
    unknown = [name for name in requested_names if name not in name_to_profile]
    if unknown:
        allowed = ", ".join(profile.name for profile in all_profiles)
        raise ValueError(f"Unknown profile(s): {unknown}. Allowed profiles: {allowed}")

    # Preserve user-provided order and remove duplicates.
    selected: list[RunProfile] = []
    seen: set[str] = set()
    for name in requested_names:
        if name in seen:
            continue
        seen.add(name)
        selected.append(name_to_profile[name])
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EfficientNetB0 7-flow reproduction")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="Dataset root with Train and Test folders")
    parser.add_argument("--train-root", default="", help="Optional explicit Train folder")
    parser.add_argument("--test-root", default="", help="Optional explicit Test folder")
    parser.add_argument("--model-dir", default="saved_models", help="Output directory for model checkpoints")
    parser.add_argument("--reports-dir", default="reports", help="Output directory for reports")
    parser.add_argument(
        "--log-path",
        default="reports/experiment_log_efficientnetb0_repro.csv",
        help="Separate CSV path for this reproduction run",
    )
    parser.add_argument(
        "--best-summary-path",
        default="reports/efficientnetb0_repro_best_run.json",
        help="Output JSON summary path for selected best run",
    )
    parser.add_argument(
        "--baseline-log",
        default="reports/experiment_log_efficientnetb0.csv",
        help="Baseline log used for comparison summary",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze-layers", type=int, default=30)
    parser.add_argument(
        "--profiles",
        default="",
        help="Comma-separated profile names to run (e.g., 05_full_train_e30). Runs all profiles if omitted.",
    )
    parser.add_argument("--append-log", action="store_true", help="Append to existing reproduction CSV")
    parser.add_argument("--no-imagenet", action="store_true", help="Disable ImageNet pre-trained weights")
    preprocessing_group = parser.add_mutually_exclusive_group()
    preprocessing_group.add_argument(
        "--include-preprocessing-layer",
        dest="include_preprocessing_layer",
        action="store_true",
        help="Enable EfficientNetB0 built-in preprocessing layer (default).",
    )
    preprocessing_group.add_argument(
        "--no-include-preprocessing-layer",
        dest="include_preprocessing_layer",
        action="store_false",
        help="Disable EfficientNetB0 built-in preprocessing layer.",
    )
    parser.set_defaults(include_preprocessing_layer=True)
    parser.add_argument(
        "--run-conversion",
        action="store_true",
        help="Run scripts/convert_tflite.py logic on selected best model",
    )
    parser.add_argument("--conversion-output-dir", default="models", help="Output directory for TFLite artifacts")
    return parser.parse_args()


def as_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def to_windows_rel(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(PROJECT_ROOT.resolve())
        return str(rel).replace("/", "\\")
    except ValueError:
        return str(path.resolve())


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
    smoke: bool,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    autotune = tf.data.AUTOTUNE

    train_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
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
    val_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
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
    test_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
        test_root,
        labels="inferred",
        label_mode="categorical",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )

    if train_ds_raw.class_names != class_names:
        raise ValueError(f"Class order mismatch. expected={class_names}, got={train_ds_raw.class_names}")

    if smoke:
        train_ds_raw = train_ds_raw.take(2)
        val_ds_raw = val_ds_raw.take(1)
        test_ds_raw = test_ds_raw.take(1)

    def preprocess_xy(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x = tf.cast(x, tf.float32)
        x = preprocess_input_efficientnet(x)
        return x, y

    train_ds = train_ds_raw.map(preprocess_xy, num_parallel_calls=autotune).ignore_errors().prefetch(autotune)
    val_ds = val_ds_raw.map(preprocess_xy, num_parallel_calls=autotune).ignore_errors().prefetch(autotune)
    test_ds = test_ds_raw.map(preprocess_xy, num_parallel_calls=autotune).ignore_errors().prefetch(autotune)
    return train_ds, val_ds, test_ds


def build_model(
    num_classes: int,
    img_size: tuple[int, int],
    use_imagenet: bool,
    include_preprocessing_layer: bool,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.10),
        ],
        name="data_augmentation",
    )

    input_tensor = tf.keras.Input(shape=(*img_size, 3), name="image")
    x = data_augmentation(input_tensor)
    base_model = EfficientNetB0(
        input_shape=(*img_size, 3),
        include_top=False,
        weights="imagenet" if use_imagenet else None,
    )
    base_model.trainable = False
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    output_tensor = tf.keras.layers.Dense(num_classes, activation="softmax", name="ripeness_probs")(x)
    model = tf.keras.Model(input_tensor, output_tensor, name="PalmRipeness_EfficientNetB0")
    return model, base_model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def evaluate_and_report(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: list[str],
    report_xlsx: Path,
) -> dict[str, Any]:
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

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    report_df = pd.DataFrame(report).transpose()
    confusion_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    with pd.ExcelWriter(report_xlsx) as writer:
        report_df.to_excel(writer, sheet_name="classification_report")
        confusion_df.to_excel(writer, sheet_name="confusion_matrix")

    return {
        "test_loss": float(test_loss),
        "accuracy": float(test_acc),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
    }


def append_log_row(log_path: Path, row: dict[str, Any], append: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row], columns=LOG_COLUMNS)
    if append and log_path.exists():
        existing = pd.read_csv(log_path)
        out_df = pd.concat([existing, row_df], ignore_index=True)
    else:
        out_df = row_df
    out_df.to_csv(log_path, index=False)


def choose_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Primary metric: accuracy. Tie-breaker: macro F1.
    best = max(rows, key=lambda r: (float(r["accuracy"]), float(r["macro_f1"])))
    return best


def read_baseline_best(baseline_log: Path) -> dict[str, Any] | None:
    if not baseline_log.exists():
        return None
    df = pd.read_csv(baseline_log)
    if "accuracy" not in df.columns or df.empty:
        return None
    best_idx = df["accuracy"].astype(float).idxmax()
    row = df.loc[best_idx].to_dict()
    return row


def maybe_convert_best(best_row: dict[str, Any], args: argparse.Namespace, train_root: Path) -> dict[str, Any] | None:
    if not args.run_conversion:
        return None

    from scripts.convert_tflite import convert_model

    best_h5 = as_project_path(str(best_row["best_model_file"]).replace("\\", "/"))
    fp32, fp16, int8, labels_path, manifest_path = convert_model(
        h5_path=str(best_h5),
        labels_path=None,
        output_dir=str(as_project_path(args.conversion_output_dir)),
        rep_data=str(train_root),
        img_size=args.img_size,
        preprocess_family="efficientnet",
    )
    return {
        "fp32": str(fp32),
        "float16": str(fp16),
        "int8": str(int8) if int8 else None,
        "labels": str(labels_path),
        "manifest": str(manifest_path),
    }


def run_profile(
    profile: RunProfile,
    args: argparse.Namespace,
    class_names: list[str],
    train_image_count: int,
    test_image_count: int,
    train_root: Path,
    test_root: Path,
    model_dir: Path,
    reports_dir: Path,
) -> dict[str, Any]:
    tf.keras.backend.clear_session()
    gc.collect()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    img_size = (args.img_size, args.img_size)

    best_model_path = model_dir / f"palm_ripeness_best_{timestamp}.h5"
    final_model_path = model_dir / f"palm_ripeness_final_{timestamp}.h5"
    finetuned_model_path = model_dir / f"palm_ripeness_finetuned_{timestamp}.h5"
    report_xlsx = reports_dir / f"classification_report_{timestamp}.xlsx"

    print(f"\n[{profile.name}] starting...")
    print(f"  mode={profile.run_mode}, epochs={profile.epochs_config}, fine_tune_epochs={profile.fine_tune_epochs}")
    print(f"  initial_lr={profile.initial_lr}, final_lr={profile.final_lr}, smoke={profile.smoke}")

    train_ds, val_ds, test_ds = make_datasets(
        train_root=train_root,
        test_root=test_root,
        class_names=class_names,
        img_size=img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        smoke=profile.smoke,
    )

    model, base_model = build_model(
        num_classes=len(class_names),
        img_size=img_size,
        use_imagenet=not args.no_imagenet,
        include_preprocessing_layer=args.include_preprocessing_layer,
    )
    compile_model(model, learning_rate=profile.initial_lr)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    warmup_epochs = 1 if profile.smoke else profile.epochs_config
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=warmup_epochs,
        callbacks=callbacks,
        verbose=1,
    )
    model.save(str(final_model_path))

    if profile.backbone_unfrozen and profile.fine_tune_epochs > 0 and not profile.smoke:
        base_model.trainable = True
        freeze_until = max(0, len(base_model.layers) - args.unfreeze_layers)
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False
        for layer in base_model.layers[freeze_until:]:
            layer.trainable = True

        compile_model(model, learning_rate=profile.final_lr)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=profile.fine_tune_epochs,
            callbacks=callbacks,
            verbose=1,
        )
        model.save(str(finetuned_model_path))

    # Keep evaluation behavior consistent with notebook (in-memory model state).
    compile_model(model, learning_rate=profile.initial_lr)
    metrics = evaluate_and_report(model, test_ds, class_names, report_xlsx)

    row = {
        "timestamp": timestamp,
        "run_mode": profile.run_mode,
        "epochs_config": profile.epochs_config,
        "fine_tune_epochs": profile.fine_tune_epochs,
        "batch_size": args.batch_size,
        "val_split": args.val_split,
        "initial_learning_rate": profile.initial_lr,
        "final_learning_rate": profile.final_lr,
        "backbone_unfrozen": profile.backbone_unfrozen,
        "train_images": train_image_count,
        "test_images": test_image_count,
        "num_classes": len(class_names),
        "accuracy": round(metrics["accuracy"], 4),
        "macro_precision": round(metrics["macro_precision"], 4),
        "macro_recall": round(metrics["macro_recall"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "best_model_file": to_windows_rel(best_model_path),
        "excel_report": to_windows_rel(report_xlsx),
        "notes": (
            f"profile={profile.name}; model=EfficientNetB0; preprocess_family=efficientnet; "
            f"include_preprocessing_layer={args.include_preprocessing_layer}; effective_warmup_epochs={warmup_epochs}"
        ),
    }
    print(
        f"[{profile.name}] accuracy={row['accuracy']:.4f}, macro_f1={row['macro_f1']:.4f}, "
        f"best_model={row['best_model_file']}"
    )
    return row


def main() -> int:
    args = parse_args()

    dataset_root = as_project_path(args.dataset_root)
    train_root = as_project_path(args.train_root) if args.train_root else dataset_root / "Train"
    test_root = as_project_path(args.test_root) if args.test_root else dataset_root / "Test"

    model_dir = as_project_path(args.model_dir)
    reports_dir = as_project_path(args.reports_dir)
    log_path = as_project_path(args.log_path)
    best_summary_path = as_project_path(args.best_summary_path)
    baseline_log = as_project_path(args.baseline_log)

    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    class_names, train_image_count, test_image_count = validate_dataset_layout(train_root, test_root)
    selected_profiles = select_profiles(PROFILES, args.profiles)

    print(f"Running EfficientNetB0 reproduction with {len(selected_profiles)} profile(s)")
    print("Selected profiles:", [profile.name for profile in selected_profiles])
    print("Dataset root:", dataset_root)
    print("Train root:", train_root)
    print("Test root:", test_root)
    print("Class names:", class_names)
    print("Train images:", train_image_count)
    print("Test images:", test_image_count)
    print("Output log:", log_path)
    print("Append mode:", bool(args.append_log))
    print("include_preprocessing_layer:", bool(args.include_preprocessing_layer))

    if log_path.exists() and not args.append_log:
        log_path.unlink()
        print("Existing reproduction log removed:", log_path)

    collected_rows: list[dict[str, Any]] = []
    for profile in selected_profiles:
        row = run_profile(
            profile=profile,
            args=args,
            class_names=class_names,
            train_image_count=train_image_count,
            test_image_count=test_image_count,
            train_root=train_root,
            test_root=test_root,
            model_dir=model_dir,
            reports_dir=reports_dir,
        )
        append_log_row(log_path, row, append=log_path.exists())
        collected_rows.append(row)

    best_row = choose_best(collected_rows)
    baseline_best = read_baseline_best(baseline_log)
    conversion_outputs = maybe_convert_best(best_row, args, train_root)

    best_summary = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "repro_log": str(log_path),
        "profiles_run": len(collected_rows),
        "selection_rule": "max accuracy, then macro_f1",
        "best_efficientnetb0": best_row,
        "baseline_best": baseline_best,
        "conversion_outputs": conversion_outputs,
    }
    best_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_summary_path, "w", encoding="utf-8") as f:
        json.dump(best_summary, f, ensure_ascii=False, indent=2)

    print("\n=== DONE ===")
    print("Best EfficientNetB0 run:", best_row)
    if baseline_best:
        print("Baseline best row:", baseline_best)
    print("Best summary saved to:", best_summary_path)

    if not args.run_conversion:
        best_h5 = best_row["best_model_file"]
        print("\nNext step (optional conversion):")
        print(
            "python scripts/convert_tflite.py "
            f"--h5 \"{best_h5}\" --rep-data \"{train_root}\" "
            f"--output-dir \"{as_project_path(args.conversion_output_dir)}\" "
            "--preprocess-family efficientnet"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
