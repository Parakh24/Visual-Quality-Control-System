import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from src.common import config

def build_model(base_model_name=config.BASE_MODEL, num_classes=config.NUM_CLASSES, input_shape=config.INPUT_SHAPE, freeze_base=config.FREEZE_LAYERS):
    """
    Builds a Transfer Learning model with a custom dense head.
    
    Args:
        base_model_name (str): Name of the base model ("MobileNetV2" or "ResNet50").
        num_classes (int): Number of output classes.
        input_shape (tuple): Shape of the input image.
        freeze_base (bool): Whether to freeze the weights of the base model.
        
    Returns:
        tf.keras.Model: The compiled model structure (uncompiled).
    """
    
    # 1. Base Model
    if base_model_name == "MobileNetV2":
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    elif base_model_name == "ResNet50":
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}")

    # Freeze base model layers if requested
    if freeze_base:
        base_model.trainable = False

    # 2. Custom Dense Head
    # Optimized for classification: reduce spatial dimensions, add non-linearity and regularization
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # First dense block
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)  # Dropout for regularization to prevent overfitting
    
    # Second dense block (optional, but good for complex features)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    # Output layer
    # Using Softmax for categorical classification (compatible with categorical_crossentropy)
    # If binary_crossentropy is preferred with a single neuron, this would be adjusted.
    # Given generators allow 'categorical', 2 neurons is robust.
    outputs = Dense(num_classes, activation='softmax')(x)

    # 3. Assemble Model
    model = Model(inputs=base_model.input, outputs=outputs, name=f"{base_model_name}_Transfer")
    
    return model

if __name__ == "__main__":
    # Smoke test
    model = build_model()
    model.summary()
