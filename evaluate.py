import tensorflow as tf
from data_loader import create_data_generators, load_data
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def evaluate_model(model):
    # Creating data generators
    train_datagen, test_datagen = create_data_generators()
    
    # Getting test generator
    _, _, test_generator = load_data(train_datagen, test_datagen)
    
    # Evaluation of the model
    test_loss, test_accuracy, test_auc = model.evaluate(test_generator)
    print(f'Test accuracy: {test_accuracy:.4f}')
    print(f'Test AUC: {test_auc:.4f}')
    
    # Generating predictions for ROC curve
    predictions = model.predict(test_generator)
    true_labels = test_generator.classes
    
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    # Plotting ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

if __name__ == "__main__":

    # Loading the trained model
    model = tf.keras.models.load_model('best_model.h5')
    evaluate_model(model)