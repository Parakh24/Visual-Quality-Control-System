import os
import numpy as np
import tensorflow as tf

from src.common import config
from src.data_pipeline.generators import create_generators


def evaluate_model(model_path):
    print(f"[INFO] Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print("[INFO] Loading test dataset...")
    _, _, test_ds = create_generators(
        data_dir=config.SPLITS_DIR,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="binary" if config.NUM_CLASSES == 2 else "categorical",
        seed=config.SEED
    )

    print("[INFO] Running evaluation...")
    results = model.evaluate(test_ds, verbose=1)
    print(f"[INFO] Test results: {results}")

    # Collect predictions for analysis
    y_true = []
    y_pred = []

    for batch_images, batch_labels in test_ds:
        preds = model.predict(batch_images)
        if config.NUM_CLASSES == 2:
            preds_classes = (preds > 0.5).astype(int).flatten()
            true_classes = batch_labels.numpy().astype(int).flatten()
        else:
            preds_classes = np.argmax(preds, axis=1)
            true_classes = np.argmax(batch_labels.numpy(), axis=1)

        y_true.extend(true_classes.tolist())
        y_pred.extend(preds_classes.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    correct_idx = np.where(y_true == y_pred)[0]
    incorrect_idx = np.where(y_true != y_pred)[0]

    print(f"[INFO] Total samples: {len(y_true)}")
    print(f"[INFO] Correct predictions: {len(correct_idx)}")
    print(f"[INFO] Incorrect predictions: {len(incorrect_idx)}")

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "correct_indices": correct_idx,
        "incorrect_indices": incorrect_idx,
        "y_true": y_true,
        "y_pred": y_pred,
    }


if __name__ == "__main__":
    model_path = os.path.join(config.TRAINED_MODELS_DIR, "vision_spec_qc.keras")
    evaluate_model(model_path)
