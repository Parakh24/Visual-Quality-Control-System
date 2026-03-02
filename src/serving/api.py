import os
import cv2
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# ---------------- CONFIG ----------------
MODEL_PATH = "models/trained/vision_spec_qc.h5"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["good", "defective"]
# ----------------------------------------

app = FastAPI(title="Visual Quality Inspection API")

# Enable CORS (optional but recommended)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create folders if not exist
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")


def preprocess_image_bytes(image_bytes: bytes):
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_norm = img_resized / 255.0
    input_tensor = np.expand_dims(img_norm, axis=0).astype(np.float32)

    return input_tensor


# ================= HOME PAGE =================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ================= PREDICT API =================
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
            "confidence": round(confidence * 100, 2)
        }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )