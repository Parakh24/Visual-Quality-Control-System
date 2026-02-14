import tensorflow as tf
import numpy as np

def generate_gradcam(model, image, last_conv_layer_name, class_index=None):
    """
    Robust Grad-CAM for Sequential CNNs.
    Splits model into:
      1) feature extractor (up to last conv layer)
      2) classifier (rest of the layers)
    This avoids 'Gradients are None' issues.
    """

    # Convert to tensor
    image = tf.convert_to_tensor(image)

    # Force build the model
    _ = model(image, training=False)

    # Find index of last conv layer
    layer_names = [layer.name for layer in model.layers]
    if last_conv_layer_name not in layer_names:
        raise ValueError(f"Layer {last_conv_layer_name} not found in model.")

    last_conv_index = layer_names.index(last_conv_layer_name)

    # Feature extractor: input -> last conv layer output
    feature_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.layers[last_conv_index].output
    )

    # Classifier: from conv output -> final prediction
    classifier_input = tf.keras.Input(shape=model.layers[last_conv_index].output.shape[1:])
    x = classifier_input
    for layer in model.layers[last_conv_index + 1:]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        conv_outputs = feature_extractor(image, training=False)
        tape.watch(conv_outputs)

        predictions = classifier_model(conv_outputs, training=False)

        if class_index is None:
            class_index = tf.argmax(predictions[0])

        class_score = predictions[:, class_index]

    # Compute gradients of class score w.r.t conv outputs
    grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        raise RuntimeError(
            "Gradients are None. This means the chosen conv layer is not suitable for Grad-CAM."
        )

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature maps
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # Normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    return heatmap.numpy()
