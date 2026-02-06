import os
import tensorflow as tf
from keras import layers
from keras.utils import image_dataset_from_directory


def get_train_augmentation():
    """
    Data augmentation pipeline for training.
    """
    return tf.keras.Sequential([
        layers.Rescaling(1.0 / 255.0),

        # Geometric augmentations
        layers.RandomRotation(0.05),        # ~10 degrees
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomZoom(0.1),

        # Photometric augmentations
        layers.RandomBrightness(0.2),

        # Flips (use carefully for PCB orientation assumptions)
        layers.RandomFlip("horizontal"),
    ])


def get_eval_preprocessing():
    """
    Preprocessing for validation/test: only normalization.
    """
    return tf.keras.Sequential([
        layers.Rescaling(1.0 / 255.0),
    ])


def create_generators(
    data_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode="categorical",
    seed=42
):
    """
    Creates train, validation, and test datasets from directory structure:

    data_dir/
        train/
            class1/
            class2/
        val/
            class1/
            class2/
        test/
            class1/
            class2/

    Returns:
        train_ds, val_ds, test_ds (tf.data.Dataset)
    """

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_ds = image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode=label_mode,
        shuffle=True,
        seed=seed
    )

    val_ds = image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode=label_mode,
        shuffle=False
    )

    test_ds = image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode=label_mode,
        shuffle=False
    )

    train_aug = get_train_augmentation()
    eval_prep = get_eval_preprocessing()

    train_ds = train_ds.map(lambda x, y: (train_aug(x, training=True), y))
    val_ds = val_ds.map(lambda x, y: (eval_prep(x, training=False), y))
    test_ds = test_ds.map(lambda x, y: (eval_prep(x, training=False), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    # Quick sanity check
    data_dir = "data/splits"
    train_ds, val_ds, test_ds = create_generators(data_dir)

    for x_batch, y_batch in train_ds.take(1):
        print("Train batch shape:", x_batch.shape, y_batch.shape)
