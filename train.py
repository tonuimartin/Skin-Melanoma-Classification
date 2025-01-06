# train.py
from config import *
from data_loader import create_data_generators, load_data
from model import create_model
import tensorflow as tf
import os
import json

def save_training_state(phase, checkpoint_path):
    """Save the current training phase to a state file."""
    state = {
        'current_phase': phase,
        'checkpoint_path': checkpoint_path
    }
    with open('training_state.json', 'w') as f:
        json.dump(state, f)

def load_training_state():
    """Load the training state if it exists."""
    try:
        with open('training_state.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def train(resume_from_checkpoint=False, force_start_phase=None):
    # Creation of data generators
    train_datagen, test_datagen = create_data_generators()
    
    # Load data
    train_generator, validation_generator, test_generator = load_data(
        train_datagen, test_datagen
    )
    
    # Create and compile model
    model, base_model = create_model()
    
    # Determine training phase and checkpoint
    training_state = load_training_state() if resume_from_checkpoint else None
    current_phase = force_start_phase or (training_state['current_phase'] if training_state else 1)
    
    # Checkpoint selection logic
    if current_phase == 1:
        checkpoint_path = 'best_first_phase.h5'
        print("Starting/Resuming First Training Phase")
    elif current_phase == 2:
        checkpoint_path = 'best_model.h5'
        print("Starting/Resuming Second Training Phase (Fine-tuning)")
    else:
        checkpoint_path = 'best_model.h5'
    
    # Loading of weights if checkpoint exists
    if os.path.exists(checkpoint_path):
        try:
            model.load_weights(checkpoint_path)
            print(f"Loaded weights from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")
    
    # First phase: Training only the top layers
    if current_phase == 1:
        # Freezing the base model for the first phase
        base_model.trainable = False
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=INITIAL_LEARNING_RATE,
                weight_decay=WEIGHT_DECAY
            ),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        # Defination of callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_first_phase.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        # First training phase
        print("Training top layers...")
        history1 = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            callbacks=callbacks
        )
        save_training_state(2, 'best_first_phase.h5')
        history2 = None
    
    # Second phase: Fine-tuning the model
    elif current_phase == 2:
        # Unfreezing the base model
        base_model.trainable = True

        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=FINE_TUNING_LEARNING_RATE,
                weight_decay=WEIGHT_DECAY
            ),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        # Defination of callbacks for second phase
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        # Second training phase
        history2 = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            callbacks=callbacks
        )
        history1 = None
        
        # Saving final model
        save_training_state(3, 'best_model.h5')
    else:
        raise ValueError("Invalid training phase")
    
    return model, history1, history2

if __name__ == "__main__":
    # Option to resume from a specific checkpoint
    model, history1, history2 = train(
        resume_from_checkpoint=True  # You can set this to False if you want to start from scratch
    )
    
    # Saving model in tflite format for mobile deployment with optimization
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        
        with open('melanoma_classifier.tflite', 'wb') as f:
            f.write(tflite_model)
        print("TFLite model saved successfully.")
    except Exception as e:
        print(f"Error converting to TFLite: {e}")