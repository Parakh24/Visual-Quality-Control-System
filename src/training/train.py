import os
import json
import tensorflow as tf
from keras import layers, models, callbacks
from keras.utils import image_dataset_from_directory

# -------------------------------
# CONFIG
# -------------------------------
DATA_DIR = "data/splits"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

MODEL_DIR = "models/trained"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# LOAD DATA
# -------------------------------
train_ds = image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

val_ds = image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# Normalize images to [0,1]
normalization_layer = layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# -------------------------------
# BUILD MODEL (Baseline CNN)
# -------------------------------
def build_model(input_shape=(224, 224, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")  # Binary classification
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

model = build_model()
model.summary()

# -------------------------------
# CALLBACKS
# -------------------------------
checkpoint_cb = callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "best_model.keras"),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

earlystop_cb = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

callback_list = [checkpoint_cb, earlystop_cb] 

# -------------------------------
# TRAIN
# -------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callback_list
)

# -------------------------------
# SAVE FINAL MODEL
# -------------------------------
model.save(os.path.join(MODEL_DIR, "vision_spec_qc.keras"))

# -------------------------------
# SAVE HISTORY (FOR error_analysis.py)
# -------------------------------
history_path = os.path.join(MODEL_DIR, "history.json")
with open(history_path, "w") as f:
    json.dump(history.history, f)

print(f"Training complete. Model and history saved to {MODEL_DIR}")

