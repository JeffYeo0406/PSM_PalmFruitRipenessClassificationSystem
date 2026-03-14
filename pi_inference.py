
import argparse, json, time, os
import numpy as np
from PIL import Image

# Prefer LiteRT (ai_edge_litert); fall back to tflite_runtime, then legacy tf.lite
try:
    from ai_edge_litert.interpreter import LiteRTInterpreter as Interpreter
    print("Using LiteRTInterpreter (ai_edge_litert)")
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
        print("Using tflite_runtime Interpreter")
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
            print("Using tf.lite.python Interpreter (legacy)")
        except ImportError as e:
            raise ImportError("No LiteRT/tflite interpreter found. Install ai-edge-litert (preferred) or tflite-runtime.") from e

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)  # [-1,1]
    return np.expand_dims(arr, axis=0)


def run_inference(model_path, labels_path, image_path, warmup=2, runs=5):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    labels = load_labels(labels_path)
    batch = load_image(image_path).astype(input_details['dtype'])

    # Warmup
    for _ in range(warmup):
        interpreter.set_tensor(input_details['index'], batch)
        interpreter.invoke()

    timings = []
    for _ in range(runs):
        start = time.time()
        interpreter.set_tensor(input_details['index'], batch)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details['index'])
        timings.append((time.time() - start) * 1000)

    probs = out[0]
    top1 = int(np.argmax(probs))
    return {
        "top1_index": top1,
        "top1_label": labels[top1] if top1 < len(labels) else str(top1),
        "top1_prob": float(probs[top1]),
        "avg_ms": float(np.mean(timings)),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="palm_ripeness_best_20260311_170850_float16.tflite", help="Path to TFLite/LiteRT model")
    parser.add_argument("--labels", default="labels_20260311_170850.json", help="Path to labels.json")
    parser.add_argument("--image", required=True, help="Path to image for inference")
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs")
    args = parser.parse_args()

    result = run_inference(args.model, args.labels, args.image, runs=args.runs)
    print(result)


if __name__ == "__main__":
    main()
