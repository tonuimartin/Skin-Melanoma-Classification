# Model parameters
IMG_HEIGHT = 256  # Increased image size for better feature detection
IMG_WIDTH = 256
BATCH_SIZE = 16  # Reduced batch size for better generalization
EPOCHS = 100  # Increased epochs with early stopping
LEARNING_RATE = 5e-5  # Lower learning rate for fine-tuning
WEIGHT_DECAY = 1e-4  # L2 regularization

# Data parameters
TRAIN_PATH = 'data/train'
TEST_PATH = 'data/test'
VALIDATION_SPLIT = 0.15  # Slightly reduced validation split

# Augmentation parameters
ROTATION_RANGE = 20
SHIFT_RANGE = 0.2
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
BRIGHTNESS_RANGE = [0.8, 1.2]