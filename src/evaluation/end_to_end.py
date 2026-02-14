import os
import time
import cv2
import numpy as np
import tensorflow as tf

# ---------------- CONFIG ----------------
MODEL_PATH = "models/trained/vision_spec_qc.keras"   # or .tflite
TEST_DIR = "data/splits/test"
IMG_SIZE = (224, 224)

# Must match folder names and training order
CLASS_NAMES = ["good", "defective"]
# ----------------------------------------


def load_model(model_path):
    if model_path.endswith(".h5") or model_path.endswith(".keras"):
        print("Loading Keras model...")
        model = tf.keras.models.load_model(model_path)
        return model, "keras"
    elif model_path.endswith(".tflite"):
        print("Loading TFLite model...")
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter, "tflite"
    else:
        raise ValueError("Unsupported model format")


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_norm = img_resized / 255.0
    input_tensor = np.expand_dims(img_norm, axis=0).astype(np.float32)
    return input_tensor


def predict(model_obj, model_type, input_tensor):
    if model_type == "keras":
        preds = model_obj.predict(input_tensor, verbose=0)
        return preds[0]
    else:
        interpreter = model_obj
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]["index"])
        return preds[0]


def main():
    model_obj, model_type = load_model(MODEL_PATH)

    total = 0
    correct = 0
    times = []

    print("Running end-to-end evaluation...")

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(TEST_DIR, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: folder not found: {class_dir}")
            continue

        for fname in os.listdir(class_dir):
            img_path = os.path.join(class_dir, fname)

            input_tensor = preprocess_image(img_path)
            if input_tensor is None:
                print(f"Skipping unreadable image: {img_path}")
                continue

            start = time.time()
            preds = predict(model_obj, model_type, input_tensor)
            end = time.time()

            pred_class = int(np.argmax(preds))

            times.append((end - start) * 1000)  # ms
            total += 1
            if pred_class == class_idx:
                correct += 1

    if total == 0:
        print("No test images found!")
        return

    accuracy = correct / total
    avg_latency = sum(times) / len(times)

    print("\n====== END-TO-END REPORT ======")
    print(f"Total samples   : {total}")
    print(f"Correct         : {correct}")
    print(f"Accuracy        : {accuracy * 100:.2f}%")
    print(f"Avg latency     : {avg_latency:.2f} ms/image")
    print("================================")


if __name__ == "__main__":
    main()
