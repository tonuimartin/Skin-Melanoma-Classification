from config import *
from data_loader import create_data_generators, load_data
from model import create_model
import tensorflow as tf

def train():
    # Create generators
    train_datagen, test_datagen = create_data_generators()
    
    # Load data
    train_generator, validation_generator, test_generator = load_data(
        train_datagen, test_datagen
    )
    
    # Create and compile model
    model = create_model()
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=5, 
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.2, 
            patience=3
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            save_best_only=True
        )
    ]
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    return model, history

if __name__ == "__main__":
    model, history = train()
    
    # Save model for mobile deployment
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open('melanoma_classifier.tflite', 'wb') as f:
        f.write(tflite_model)