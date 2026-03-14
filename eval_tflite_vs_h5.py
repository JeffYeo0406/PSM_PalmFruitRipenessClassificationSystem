import argparse
import json
import os
import time
from typing import List, Tuple

import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

try:
    from ai_edge_litert.interpreter import LiteRTInterpreter as Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

IMG_SIZE = (224, 224)
DEFAULT_DATA_DIR = r"C:\\Users\\jeffy\\iCloudDrive\\College\\Y4S1\\BERN4973 Final Year Project PSM1\\Assignment\\Dataset\\Dataset1\\Test"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path: str) -> str:
    """Resolve a possibly relative path, preferring cwd then this script's directory."""
    if os.path.isabs(path):
        return path
    # 1) relative to current working directory
    if os.path.exists(path):
        return path
    # 2) relative to the script directory
    alt = os.path.join(SCRIPT_DIR, path)
    if os.path.exists(alt):
        return alt
    # 3) if path contains nested folder (e.g., "saved_models/foo" while cwd is already saved_models)
    base = os.path.basename(path)
    alt2 = os.path.join(SCRIPT_DIR, base)
    return alt2 if os.path.exists(alt2) else path


def load_labels(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_samples(root: str, labels: List[str]) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []
    for idx, label in enumerate(labels):
        class_dir = os.path.join(root, label)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            full = os.path.join(class_dir, fname)
            if os.path.isfile(full):
                samples.append((full, idx))
    return samples


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict_h5(model, batch: np.ndarray) -> np.ndarray:
    return model.predict(batch, verbose=0)


def predict_tflite(interpreter, input_index: int, output_index: int, batch: np.ndarray) -> np.ndarray:
    interpreter.set_tensor(input_index, batch)
    interpreter.invoke()
    return interpreter.get_tensor(output_index)


def evaluate(models_root: str, data_dir: str, labels_path: str, h5_path: str, tflite_path: str):
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 model not found: {h5_path}")
    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"TFLite model not found: {tflite_path}")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    labels = load_labels(labels_path)
    samples = discover_samples(data_dir, labels)
    if not samples:
        raise RuntimeError(f"No samples found under {data_dir}; ensure subfolders match labels: {labels}")

    print(f"Found {len(samples)} images across {len(labels)} classes")

    h5_model = load_model(h5_path)

    tflite_interpreter = Interpreter(model_path=tflite_path)
    tflite_interpreter.allocate_tensors()
    input_details = tflite_interpreter.get_input_details()[0]
    output_details = tflite_interpreter.get_output_details()[0]

    y_true: List[int] = []
    y_pred_h5: List[int] = []
    y_pred_tflite: List[int] = []

    tflite_times: List[float] = []
    h5_times: List[float] = []

    for path, label_idx in samples:
        batch = load_image(path)

        start = time.time()
        logits_h5 = predict_h5(h5_model, batch)
        h5_times.append((time.time() - start) * 1000)

        batch_tflite = batch.astype(input_details["dtype"])
        start = time.time()
        logits_tflite = predict_tflite(tflite_interpreter, input_details["index"], output_details["index"], batch_tflite)
        tflite_times.append((time.time() - start) * 1000)

        y_true.append(label_idx)
        y_pred_h5.append(int(np.argmax(logits_h5)))
        y_pred_tflite.append(int(np.argmax(logits_tflite)))

    try:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    except ImportError:
        accuracy_score = classification_report = confusion_matrix = None

    def summarize(name: str, preds: List[int]):
        correct = sum(int(p == t) for p, t in zip(preds, y_true))
        acc = correct / len(y_true)
        print(f"\n{name} accuracy: {acc:.4f} ({correct}/{len(y_true)})")
        if accuracy_score:
            print(classification_report(y_true, preds, target_names=labels, digits=4))
            print("Confusion matrix:\n", confusion_matrix(y_true, preds))

    summarize("H5", y_pred_h5)
    summarize("TFLite", y_pred_tflite)

    print(f"\nAvg latency (ms): H5={np.mean(h5_times):.2f}, TFLite={np.mean(tflite_times):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Compare H5 and TFLite models on a labeled image folder")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="Root folder with subfolders per class, e.g., images/test")
    parser.add_argument("--labels", default="labels_20260311_170850.json", help="Path to labels JSON")
    parser.add_argument("--h5", default="palm_ripeness_best_20260311_170850.h5", help="Path to Keras H5 model")
    parser.add_argument("--tflite", default="palm_ripeness_best_20260311_170850_float16.tflite", help="Path to TFLite model")
    args = parser.parse_args()
    data_dir = resolve_path(args.data_dir)
    labels_path = resolve_path(args.labels)
    h5_path = resolve_path(args.h5)
    tflite_path = resolve_path(args.tflite)

    evaluate(
        models_root=os.path.dirname(os.path.abspath(h5_path)),
        data_dir=data_dir,
        labels_path=labels_path,
        h5_path=h5_path,
        tflite_path=tflite_path,
    )


if __name__ == "__main__":
    main()
