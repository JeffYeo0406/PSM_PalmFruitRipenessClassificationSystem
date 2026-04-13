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
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def load_interpreter(model_path):
    """Load TFLite interpreter with fallback chain."""
    try:
        from ai_edge_litert.interpreter import LiteRTInterpreter as Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter
    return Interpreter(model_path=model_path)

def load_labels(labels_path):
    with open(labels_path, "r") as f:
        return json.load(f)

def preprocess_image(image_path, img_size=224):
    """Load and preprocess single image."""
    img = tf.io.read_file(str(image_path))
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = preprocess_input(img)
    return np.expand_dims(img.numpy(), axis=0).astype(np.float32)

def evaluate_model(interpreter, dataset_dir, labels, img_size=224):
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
                input_data = preprocess_image(img_path, img_size)
                
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
    args = parser.parse_args()
    
    # Load labels
    labels = load_labels(args.labels)
    print(f"Classes: {labels}")
    
    # Load interpreters
    print("\nLoading FP32 model...")
    fp32_interpreter = load_interpreter(args.model_fp32)
    fp32_interpreter.allocate_tensors()
    
    print("Loading INT8 model...")
    int8_interpreter = load_interpreter(args.model_int8)
    int8_interpreter.allocate_tensors()
    
    # Evaluate FP32
    print("\nEvaluating FP32 model...")
    fp32_acc, fp32_correct, fp32_total = evaluate_model(
        fp32_interpreter, args.data_dir, labels, args.img_size
    )
    print(f"FP32 Accuracy: {fp32_acc:.4f} ({fp32_correct}/{fp32_total})")
    
    # Evaluate INT8
    print("\nEvaluating INT8 model...")
    int8_acc, int8_correct, int8_total = evaluate_model(
        int8_interpreter, args.data_dir, labels, args.img_size
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