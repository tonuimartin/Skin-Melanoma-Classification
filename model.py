import tensorflow as tf
from keras import layers, models
from keras.applications import EfficientNetB4  # Changed to EfficientNet
from config import *

def create_model():
    # Use EfficientNetB4 instead of ResNet50
    base_model = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    # Freeze the base model first
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(1024, activation='swish'),  # Increased capacity, using swish
        layers.Dropout(0.5),
        layers.BatchNormalization(),
        layers.Dense(512, activation='swish'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(256, activation='swish'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Custom learning rate schedule with warmup
    initial_learning_rate = LEARNING_RATE
    warmup_epochs = 5
    decay_epochs = 15
    
    def warmup_cosine_decay_schedule(epoch):
        if epoch < warmup_epochs:
            return initial_learning_rate * ((epoch + 1) / warmup_epochs)
        else:
            decay_progress = (epoch - warmup_epochs) / (decay_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1 + tf.cos(tf.constant(3.14159) * decay_progress))
            return initial_learning_rate * cosine_decay
    
    # Compile with additional metrics
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=initial_learning_rate,
            weight_decay=WEIGHT_DECAY
        ),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )
    
    return model, warmup_cosine_decay_schedule