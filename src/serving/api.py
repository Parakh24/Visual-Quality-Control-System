import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# ---------------- CONFIG ----------------
MODEL_PATH = "models/trained/vision_spec_qc.keras"  # change if your file is .h5
IMG_SIZE = (224, 224)
CLASS_NAMES = ["good", "defective"]  # must match your folders / training order
# ---------------------------------------

app = FastAPI(title="Visual Quality Inspection API")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")


def preprocess_image_bytes(image_bytes: bytes):
    # Convert bytes to numpy image
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_norm = img_resized / 255.0
    input_tensor = np.expand_dims(img_norm, axis=0).astype(np.float32)
    return input_tensor


@app.get("/")
def root():
    return {"message": "Visual Quality Inspection API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image_bytes(image_bytes)

        preds = model.predict(input_tensor, verbose=0)[0]
        pred_class = int(np.argmax(preds))
        confidence = float(np.max(preds))

        result = {
            "class": CLASS_NAMES[pred_class],
            "confidence": confidence
        }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
