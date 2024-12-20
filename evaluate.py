import tensorflow as tf
from data_loader import create_data_generators, load_data

def evaluate_model(model):
    _, test_datagen = create_data_generators()
    _, _, test_generator = load_data(test_datagen)
    
    # Evaluate the model
    test_loss, test_accuracy, test_auc = model.evaluate(test_generator)
    print(f'Test accuracy: {test_accuracy:.4f}')
    print(f'Test AUC: {test_auc:.4f}')