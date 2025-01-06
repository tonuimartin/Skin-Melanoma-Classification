from keras.preprocessing.image import ImageDataGenerator
from config import *

def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=ROTATION_RANGE,
        width_shift_range=SHIFT_RANGE,
        height_shift_range=SHIFT_RANGE,
        zoom_range=ZOOM_RANGE,
        horizontal_flip=HORIZONTAL_FLIP,
        vertical_flip=VERTICAL_FLIP,
        fill_mode='reflect',  
        brightness_range=BRIGHTNESS_RANGE,
        validation_split=VALIDATION_SPLIT,
        preprocessing_function=None  
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, test_datagen

def load_data(train_datagen, test_datagen):
    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=True
    )
    
    test_generator = test_datagen.flow_from_directory(
        TEST_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False  
    )
    
    return train_generator, validation_generator, test_generator