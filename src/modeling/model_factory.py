from keras import layers, models, optimizers
from keras.applications import MobileNetV2, ResNet50


def build_model(base_model_name, input_shape, num_classes, learning_rate, freeze_layers=True):
    """
    Build and compile a Keras model based on the specified backbone.
    """

    if base_model_name == "MobileNetV2":
        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name == "ResNet50":
        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}")

    # Freeze backbone layers if requested
    if freeze_layers:
        for layer in base_model.layers:
            layer.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        loss = "categorical_crossentropy"

    model = models.Model(inputs=base_model.input, outputs=outputs)

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"]
    )

    return model
