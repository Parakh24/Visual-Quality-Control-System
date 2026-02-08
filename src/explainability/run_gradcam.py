import tensorflow as tf
from keras.applications import MobileNetV2
from src.explainability.gradcam import generate_gradcam, load_and_preprocess_image

# Pretrained model (NO local file needed)
model = MobileNetV2(weights="imagenet")

image = load_and_preprocess_image(
    "data/splits/test/defective/00041005_test.jpg"
)


heatmap = generate_gradcam(
    model=model,
    image=image,
    last_conv_layer_name="Conv_1"
)

print("Heatmap shape:", heatmap.shape)
print("Min / Max:", heatmap.min(), heatmap.max())
