from keras.preprocessing.image import ImageDataGenerator
from config import *

def create_data_generators():
    train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,  # Increase rotation range
    width_shift_range=0.3,  # Increase shift range
    height_shift_range=0.3,
    shear_range=0.2,  # Add shearing
    zoom_range=0.2,   # Add zooming
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range=[0.7,1.3],  # Add brightness variation
    validation_split=VALIDATION_SPLIT
)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, test_datagen

def load_data(train_datagen, test_datagen):
    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )
    
    test_generator = test_datagen.flow_from_directory(
        TEST_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    return train_generator, validation_generator, test_generator