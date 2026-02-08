import os
import json
import tensorflow as tf

from ..common import config


try:
    from ..common.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

from ..modeling.model_factory import build_model
from ..data_pipeline.generators import create_generators
from ..training.callbacks import get_callbacks


def save_history(history, path):
    history_dict = history.history
    with open(path, "w") as f:
        json.dump(history_dict, f, indent=4)
    logger.info(f"Training history saved to: {path}")


def main():
    logger.info("Starting training pipeline...")

    # Check GPU availability
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info(f"GPU detected: {gpus}")
    else:
        logger.warning("No GPU detected. Training will run on CPU.")

    # Load data generators
    logger.info("Loading data generators...")
    train_ds, val_ds, _ = create_generators(
        data_dir=config.SPLITS_DIR,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="binary" if config.NUM_CLASSES == 2 else "categorical",
        seed=config.SEED
    )

    # Build model
    logger.info(f"Building model with base: {config.BASE_MODEL}")
    model = build_model(
        base_model_name=config.BASE_MODEL,
        input_shape=config.INPUT_SHAPE,
        num_classes=config.NUM_CLASSES,
        learning_rate=config.LEARNING_RATE,
        freeze_layers=config.FREEZE_LAYERS
    )

    model.summary(print_fn=logger.info)

    
    logger.info("Setting up callbacks...")
    callbacks = get_callbacks(config.BASE_MODEL)

    # Train model
    logger.info("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks
    )

    # Save final model
    final_model_path = os.path.join(config.TRAINED_MODELS_DIR, "vision_spec_qc.h5")
    model.save(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")

    # Save training history
    history_path = os.path.join(config.TRAINED_MODELS_DIR, "history.json")
    save_history(history, history_path)

    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()
