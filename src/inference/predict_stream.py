import cv2
import numpy as np
import tensorflow as tf

# 🔹 ADD THIS IMPORT
from tensorflow.keras.applications import MobileNetV2

# 🔹 Grad-CAM imports
from src.explainability.gradcam import generate_gradcam
from src.explainability.visualize import overlay_heatmap


# =========================
# MODEL LOADING (HERE)
# =========================

# 🔹 USE PRETRAINED MODEL (Week-3 & Week-4)
model = MobileNetV2(weights="imagenet")

# 🔹 Last convolutional layer for Grad-CAM
LAST_CONV_LAYER = "Conv_1"


# =========================
# PREPROCESSING FUNCTION
# =========================

def preprocess_frame(frame, img_size=(224, 224)):
    frame_resized = cv2.resize(frame, img_size)
    frame_resized = frame_resized.astype("float32") / 255.0
    return np.expand_dims(frame_resized, axis=0)


# =========================
# LIVE CAMERA INFERENCE
# =========================

def run_live_inference():
    cap = cv2.VideoCapture(0)  # Webcam

    if not cap.isOpened():
        print("❌ Cannot access camera")
        return

    print("✅ Camera started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess_frame(frame)

        # 🔹 Prediction
        preds = model.predict(input_tensor, verbose=0)
        pred_class = np.argmax(preds)

        # 🔹 Grad-CAM
        heatmap = generate_gradcam(
            model,
            input_tensor,
            LAST_CONV_LAYER,
            class_index=pred_class
        )

        # 🔹 Overlay heatmap
        overlay = overlay_heatmap(frame, heatmap)

        # 🔹 Display label
        label = f"Predicted Class: {pred_class}"
        cv2.putText(
            overlay,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.imshow("Live QC Inference + Grad-CAM", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    run_live_inference()
