import tensorflow as tf
import numpy as np
import cv2


def load_and_preprocess_image(image_path, img_size=(224, 224)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img = cv2.resize(img, img_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def generate_gradcam(model, image, last_conv_layer_name, class_index=None):
    """
    Generate Grad-CAM heatmap for a given image.
    
    Args:
        model: Trained Keras model
        image: Preprocessed image (shape: 1, H, W, C)
        last_conv_layer_name: Name of last convolutional layer in the model
        class_index: Optional class index to explain
    Returns:
        heatmap (H, W) numpy array in range [0, 1]
    """

    # Build a model that maps the input image to the activations
    # of the last conv layer and the final predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)

        if class_index is None:
            class_index = tf.argmax(predictions[0])

        loss = predictions[:, class_index]

    # Compute gradients of the target class w.r.t. conv outputs
    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the convolution outputs
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    return heatmap.numpy()
