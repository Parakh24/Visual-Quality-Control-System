import os

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
TRAIN_DIR = os.path.join(SPLITS_DIR, "train")
VAL_DIR = os.path.join(SPLITS_DIR, "val")
TEST_DIR = os.path.join(SPLITS_DIR, "test")

# Artifacts paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
TRAINED_MODELS_DIR = os.path.join(MODELS_DIR, "trained")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Data parameters
IMG_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
NUM_CLASSES = 2  # Binary classification (Pass/Defect)

# Model parameters
# Supported base models: "MobileNetV2", "ResNet50"
BASE_MODEL = "MobileNetV2"
FREEZE_LAYERS = True

# Training parameters
LEARNING_RATE = 1e-4
EPOCHS = 20
PATIENCE = 5  # For early stopping
SEED = 42
