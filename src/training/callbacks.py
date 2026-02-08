import os
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
from src.common import config


def get_callbacks(model_name):
    """
    Creates a list of callbacks for training.
    """

    callbacks = []

    # 1. Model Checkpoint
    checkpoint_path = os.path.join(config.TRAINED_MODELS_DIR, f"{model_name}_best.h5")
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min"
    )
    callbacks.append(checkpoint)

    # 2. Early Stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=config.PATIENCE,
        verbose=1,
        restore_best_weights=True
    )
    callbacks.append(early_stopping)

    # 3. Reduce LR on Plateau
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(reduce_lr)

    # 4. CSV Logger
    log_path = os.path.join(config.LOGS_DIR, f"{model_name}_training_log.csv")
    csv_logger = CSVLogger(log_path, append=False)
    callbacks.append(csv_logger)

    # 5. TensorBoard
    tensorboard_log_dir = os.path.join(config.LOGS_DIR, "tensorboard", model_name)
    tensorboard = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)
    callbacks.append(tensorboard)

    return callbacks
