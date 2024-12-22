from config import *
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from data_loader import create_data_generators, load_data
from model import create_model
import tensorflow as tf
import tensorflow_addons as tfa

def train():
    # Create generators
    train_datagen, test_datagen = create_data_generators()
    
    # Load data
    train_generator, validation_generator, test_generator = load_data(
        train_datagen, test_datagen
    )
    
    # Create and compile model
    model, lr_schedule = create_model()
    
    # Enhanced callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',  # Monitor AUC instead of loss
            patience=10,  # Increased patience
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,  # More gentle reduction
            patience=5,
            min_lr=1e-6,
            mode='max'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.LearningRateScheduler(lr_schedule),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]
    
    # Calculate class weights
    labels = []
    train_generator.reset()
    
    num_samples = train_generator.samples
    num_batches = int(np.ceil(num_samples / BATCH_SIZE))
    
    for _ in range(num_batches):
        _, batch_labels = next(train_generator)
        labels.extend(batch_labels)
    
    labels = np.array(labels)
    
    # Enhanced class weights calculation
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    # Adjust class weights to be less extreme
    max_weight = max(class_weight_dict.values())
    class_weight_dict = {k: v/max_weight * 2 for k, v in class_weight_dict.items()}
    
    train_generator.reset()
    
    # Two-phase training
    # Phase 1: Train only the top layers
    history1 = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )
    
    # Phase 2: Fine-tune the whole model
    for layer in model.layers[0].layers[-50:]:  # Unfreeze last 50 layers
        layer.trainable = True
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=LEARNING_RATE/10,
            weight_decay=WEIGHT_DECAY
        ),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    history2 = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        initial_epoch=10
    )
    
    return model, history1, history2

if __name__ == "__main__":
    model, history1, history2 = train()
    
    # Enhanced TFLite conversion with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    
    with open('melanoma_classifier.tflite', 'wb') as f:
        f.write(tflite_model)