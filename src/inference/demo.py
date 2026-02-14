import os
import cv2
import time
import numpy as np
import tensorflow as tf

from src.explainability.gradcam import generate_gradcam
from src.explainability.visualize import overlay_heatmap

# ---------------- CONFIG ----------------
MODEL_PATH = "models/trained/vision_spec_qc.keras"
OUTPUT_DIR = "assets/sample_outputs"
IMG_SIZE = (224, 224)

# Must match your folder names & training order
CLASS_NAMES = ["good", "defective"]

# IMPORTANT: must match model.summary()
LAST_CONV_LAYER_NAME = "conv2d_2"

TEST_ROOT = "data/splits/test"
# ----------------------------------------


def find_any_test_image(root):
    for root_dir, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                return os.path.join(root_dir, f)
    return None


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
        raise ValueError(f"Could not read image: {img_path}")

    original = img.copy()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_norm = img_resized / 255.0
    input_tensor = np.expand_dims(img_norm, axis=0).astype(np.float32)
    return original, input_tensor


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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find a real test image automatically
    image_path = find_any_test_image(TEST_ROOT)
    if image_path is None:
        raise RuntimeError("No test images found in data/splits/test/")

    print(f"Using test image: {image_path}")

    # Load model
    model_obj, model_type = load_model(MODEL_PATH)

    # Print summary once to verify layer names
    model_obj.summary()

    # Load & preprocess image
    original_img, input_tensor = preprocess_image(image_path)

    # Inference
    start = time.time()
    preds = predict(model_obj, model_type, input_tensor)
    end = time.time()

    pred_class = int(np.argmax(preds))
    confidence = float(np.max(preds))

    label = CLASS_NAMES[pred_class]
    print(f"Prediction: {label} (confidence={confidence:.4f})")
    print(f"Inference time: {(end - start) * 1000:.2f} ms")

    # Grad-CAM only works with Keras model
    if model_type == "keras":
        heatmap = generate_gradcam(
    model=model_obj,
    image=input_tensor,   # numpy array
    last_conv_layer_name=LAST_CONV_LAYER_NAME,
    class_index=pred_class
)


        overlay = overlay_heatmap(original_img, heatmap)

        # Put label on image
        text = f"{label.upper()} ({confidence:.2f})"
        cv2.putText(overlay, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        output_path = os.path.join(OUTPUT_DIR, "demo_result.jpg")
        cv2.imwrite(output_path, overlay)
        print(f"Saved demo output to: {output_path}")

        # Show result
        cv2.imshow("Demo Result", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Grad-CAM skipped (TFLite model).")


if __name__ == "__main__":
    main()
