"""
Validate TFLite model accuracy against a test dataset.
Compares INT8 vs FP32 to ensure accuracy drop < 2%.
"""

import argparse
import json
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobilenet_v2
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_input_mobilenet_v3


SUPPORTED_PREPROCESS_FAMILIES = ("mobilenet_v2", "mobilenet_v3", "none")

def load_interpreter(model_path, prefer_reference_kernels=False):
    """Load TFLite interpreter with fallback chain.

    If prefer_reference_kernels=True, force TensorFlow reference kernels without
    delegates for maximum compatibility.
    """
    if not prefer_reference_kernels:
        try:
            from ai_edge_litert.interpreter import LiteRTInterpreter as Interpreter
            return Interpreter(model_path=model_path)
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
                return Interpreter(model_path=model_path)
            except ImportError:
                pass

    from tensorflow.lite.python.interpreter import Interpreter
    if prefer_reference_kernels:
        return Interpreter(
            model_path=model_path,
            experimental_delegates=[],
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
            num_threads=1,
        )
    return Interpreter(model_path=model_path)

def load_labels(labels_path):
    with open(labels_path, "r") as f:
        return json.load(f)

def _apply_preprocess(img, preprocess_family="mobilenet_v2"):
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


def preprocess_image(image_path, img_size=224, preprocess_family="mobilenet_v2"):
    """Load and preprocess single image."""
    img = tf.io.read_file(str(image_path))
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = _apply_preprocess(img, preprocess_family)
    return np.expand_dims(img.numpy(), axis=0).astype(np.float32)


def evaluate_model(interpreter, dataset_dir, labels, img_size=224, preprocess_family="mobilenet_v2"):
    """Evaluate model on dataset directory."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Handle quantization if needed
    floating_model = input_details[0]['dtype'] == np.float32
    
    correct = 0
    total = 0
    
    for class_idx, class_name in enumerate(labels):
        class_dir = Path(dataset_dir) / class_name
        if not class_dir.exists():
            continue
            
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
        
        for img_path in tqdm(image_files, desc=f"Evaluating {class_name}", leave=False):
            try:
                # Preprocess
                input_data = preprocess_image(
                    img_path,
                    img_size,
                    preprocess_family=preprocess_family,
                )
                
                # Quantize if needed
                if not floating_model:
                    input_scale, input_zero_point = input_details[0]["quantization"]
                    if input_scale != 0:
                        input_data = (input_data / input_scale + input_zero_point).astype(input_details[0]["dtype"])
                
                # Run inference
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                
                # Dequantize if needed
                if not floating_model:
                    output_scale, output_zero_point = output_details[0]["quantization"]
                    if output_scale != 0:
                        output_data = (output_data - output_zero_point) * output_scale
                
                # Get prediction
                pred_idx = np.argmax(output_data[0])
                if pred_idx == class_idx:
                    correct += 1
                total += 1
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total

def main():
    parser = argparse.ArgumentParser(description="Validate TFLite model accuracy")
    parser.add_argument("--model-fp32", required=True, help="Path to FP32 TFLite model")
    parser.add_argument("--model-int8", required=True, help="Path to INT8 TFLite model")
    parser.add_argument("--labels", required=True, help="Path to labels.json")
    parser.add_argument("--data-dir", required=True, help="Path to test/validation dataset")
    parser.add_argument("--img-size", type=int, default=224, help="Image size")
    parser.add_argument(
        "--preprocess-family",
        default="mobilenet_v2",
        choices=SUPPORTED_PREPROCESS_FAMILIES,
        help="Input preprocessing family used during model training/conversion.",
    )
    args = parser.parse_args()
    
    # Load labels
    labels = load_labels(args.labels)
    print(f"Classes: {labels}")
    print(f"Preprocess family: {args.preprocess_family}")
    
    # Load interpreters
    print("\nLoading FP32 model...")
    fp32_interpreter = load_interpreter(args.model_fp32)
    try:
        fp32_interpreter.allocate_tensors()
    except RuntimeError as exc:
        print(f"FP32 interpreter allocation failed with default delegate chain: {exc}")
        print("Retrying FP32 with TensorFlow reference kernels (no delegate)...")
        fp32_interpreter = load_interpreter(args.model_fp32, prefer_reference_kernels=True)
        fp32_interpreter.allocate_tensors()
    
    print("Loading INT8 model...")
    int8_interpreter = load_interpreter(args.model_int8)
    try:
        int8_interpreter.allocate_tensors()
    except RuntimeError as exc:
        print(f"INT8 interpreter allocation failed with default delegate chain: {exc}")
        print("Retrying INT8 with TensorFlow reference kernels (no delegate)...")
        int8_interpreter = load_interpreter(args.model_int8, prefer_reference_kernels=True)
        int8_interpreter.allocate_tensors()
    
    # Evaluate FP32
    print("\nEvaluating FP32 model...")
    fp32_acc, fp32_correct, fp32_total = evaluate_model(
        fp32_interpreter,
        args.data_dir,
        labels,
        args.img_size,
        preprocess_family=args.preprocess_family,
    )
    print(f"FP32 Accuracy: {fp32_acc:.4f} ({fp32_correct}/{fp32_total})")
    
    # Evaluate INT8
    print("\nEvaluating INT8 model...")
    int8_acc, int8_correct, int8_total = evaluate_model(
        int8_interpreter,
        args.data_dir,
        labels,
        args.img_size,
        preprocess_family=args.preprocess_family,
    )
    print(f"INT8 Accuracy: {int8_acc:.4f} ({int8_correct}/{int8_total})")
    
    # Calculate drop
    acc_drop = fp32_acc - int8_acc
    acc_drop_pct = (acc_drop / fp32_acc) * 100 if fp32_acc > 0 else 0
    
    print("\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)
    print(f"FP32 Accuracy: {fp32_acc:.4f}")
    print(f"INT8 Accuracy: {int8_acc:.4f}")
    print(f"Absolute Drop: {acc_drop:.4f}")
    print(f"Relative Drop: {acc_drop_pct:.2f}%")
    
    if acc_drop_pct < 2.0:
        print("\n✅ PASS: INT8 accuracy drop < 2%")
        return 0
    else:
        print(f"\n❌ FAIL: INT8 accuracy drop >= 2% ({acc_drop_pct:.2f}%)")
        return 1

if __name__ == "__main__":
    exit(main())