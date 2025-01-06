# config.py
# Model parameters
IMG_HEIGHT = 299  # Increased for better feature detection
IMG_WIDTH = 299
BATCH_SIZE = 16  # Smaller batch size for better generalization
EPOCHS = 150  # More epochs with early stopping
INITIAL_LEARNING_RATE = 1e-3  # Higher initial learning rate
FINE_TUNING_LEARNING_RATE = 1e-5  # Separate learning rate for fine-tuning
WEIGHT_DECAY = 2e-4  # Increased L2 regularization

# Data parameters
TRAIN_PATH = 'data/train'
TEST_PATH = 'data/test'
VALIDATION_SPLIT = 0.2  # Reduced validation split for more training data

# Augmentation parameters (more conservative)
ROTATION_RANGE = 15
SHIFT_RANGE = 0.15
ZOOM_RANGE = 0.15
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
BRIGHTNESS_RANGE = [0.9, 1.1]