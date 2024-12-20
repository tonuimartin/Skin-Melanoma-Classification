import tensorflow as tf
from data_loader import create_data_generators, load_data

def evaluate_model(model):
    # Create both generators even though we only need test
    train_datagen, test_datagen = create_data_generators()
    
    # Get all generators but we'll only use test
    _, _, test_generator = load_data(train_datagen, test_datagen)
    
    # Evaluate the model
    test_loss, test_accuracy, test_auc = model.evaluate(test_generator)
    print(f'Test accuracy: {test_accuracy:.4f}')
    print(f'Test AUC: {test_auc:.4f}')

if __name__ == "__main__":
    # Load the trained model
    model = tf.keras.models.load_model('best_model.h5')
    evaluate_model(model)