import tensorflow as tf
import numpy as np
from PIL import Image
import os
from config import IMG_HEIGHT, IMG_WIDTH
import argparse
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

class MelanomaPredictor:
    def __init__(self, model_path='best_model.h5'):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ['benign', 'malignant']
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for prediction.
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            # Loading and resizing image
            img = Image.open(image_path)
            img = img.resize((IMG_WIDTH, IMG_HEIGHT))
            img = img.convert('RGB')
            
            # Converting to array and preprocessing
            img_array = np.array(img)
            img_array = img_array.astype('float32')
            img_array /= 255.0  
            
            return np.expand_dims(img_array, axis=0)
        
        except Exception as e:
            raise Exception(f"Error preprocessing image {image_path}: {str(e)}")
    
    def predict_single(self, image_path):
        """
        Make prediction on a single image.
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            dict: Prediction results including class and confidence
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Preprocessing image
            preprocessed_img = self.preprocess_image(image_path)
            
            prediction = self.model.predict(preprocessed_img)[0][0]
            
            # Getting confidence and predicted class
            confidence = prediction if prediction >= 0.5 else 1 - prediction
            predicted_class = self.class_names[1] if prediction >= 0.5 else self.class_names[0]
            
            return {
                'filename': os.path.basename(image_path),
                'predicted_class': predicted_class,
                'confidence': float(confidence * 100),  
                'raw_prediction': float(prediction)
            }
            
        except Exception as e:
            raise Exception(f"Error predicting image {image_path}: {str(e)}")
    
    def predict_batch(self, image_dir, supported_formats=('.jpg', '.jpeg', '.png')):
        """
        Make predictions on all images in a directory.
        
        Args:
            image_dir (str): Directory containing images
            supported_formats (tuple): Supported image file extensions
        
        Returns:
            list: List of prediction results for each image
        """
        results = []
        
        try:
            # Getting all image files in the directory
            image_files = [
                f for f in Path(image_dir).iterdir()
                if f.suffix.lower() in supported_formats
            ]
            
            if not image_files:
                raise Exception(f"No supported images found in {image_dir}")
            
            # Processing each image
            for image_path in image_files:
                try:
                    result = self.predict_single(str(image_path))
                    results.append(result)
                except Exception as e:
                    print(f"Warning: Failed to process {image_path}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            raise Exception(f"Error processing directory {image_dir}: {str(e)}")

    def evaluate_test_directory(self, test_dir, supported_formats=('.jpg', '.jpeg', '.png')):
        """
        Evaluate model on a test directory with 'benign' and 'malignant' subdirectories.
        Creates confusion matrix and prints detailed metrics.
        
        Args:
            test_dir (str): Path to test directory containing class subdirectories
            supported_formats (tuple): Supported image file extensions
        
        Returns:
            tuple: (results_list, classification_report_dict, confusion_matrix_array)
        """
        true_labels = []
        predicted_labels = []
        raw_predictions = []
        results_list = []
        
        try:
            # Processing of each class directory
            for class_name in self.class_names:
                class_dir = os.path.join(test_dir, class_name)
                if not os.path.exists(class_dir):
                    raise Exception(f"Class directory not found: {class_dir}")
                
                # Getting all images in the class directory
                image_files = [
                    f for f in Path(class_dir).iterdir()
                    if f.suffix.lower() in supported_formats
                ]
                
                # Processing each image in the directory
                for image_path in image_files:
                    try:
                        result = self.predict_single(str(image_path))
                        results_list.append(result)
                        
                        true_labels.append(class_name)
                        predicted_labels.append(result['predicted_class'])
                        raw_predictions.append(result['raw_prediction'])
                        
                    except Exception as e:
                        print(f"Warning: Failed to process {image_path}: {str(e)}")
                        continue
            
            # Convert labels to binary for ROC curve
            y_true = [1 if label == 'malignant' else 0 for label in true_labels]
            
            # Calculate ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_true, raw_predictions)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.savefig('roc_curve.png')
            plt.close()
            
            # Calculation of metrics
            conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=self.class_names)
            
            # Plotting confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig('confusion_matrix.png')
            plt.close()
            
            # Calculate combined metrics
            tn, fp, fn, tp = conf_matrix.ravel()
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            print("\nOverall Model Metrics:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Sensitivity/Recall: {sensitivity:.4f}")
            print(f"Specificity: {specificity:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"F1 Score: {f1_score:.4f}")
            print(f"AUC-ROC: {roc_auc:.4f}")
            
            print("\nConfusion Matrix Breakdown:")
            print(f"True Negatives (Correct Benign): {tn}")
            print(f"False Positives (Incorrect Malignant): {fp}")
            print(f"False Negatives (Incorrect Benign): {fn}")
            print(f"True Positives (Correct Malignant): {tp}")
            
            return results_list, conf_matrix, {'accuracy': accuracy, 
                                             'sensitivity': sensitivity,
                                             'specificity': specificity,
                                             'precision': precision,
                                             'f1_score': f1_score,
                                             'auc_roc': roc_auc}
            
        except Exception as e:
            raise Exception(f"Error evaluating test directory: {str(e)}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Predict melanoma from images')
    parser.add_argument('input_path', help='Path to image file or directory')
    parser.add_argument('--model', default='best_model.h5', help='Path to model file')
    parser.add_argument('--batch', action='store_true', help='Process entire directory')
    parser.add_argument('--evaluate', action='store_true', 
                       help='Evaluate test directory with class subdirectories')
    
    args = parser.parse_args()
    
    try:
        predictor = MelanomaPredictor(args.model)
        
        if args.evaluate:
            print("\nPerforming Test Directory Evaluation:")
            results, _, _ = predictor.evaluate_test_directory(args.input_path)
            print("\nDetailed Predictions:")
            for result in results:
                print(f"\nFile: {result['filename']}")
                print(f"Prediction: {result['predicted_class']}")
                print(f"Confidence: {result['confidence']:.2f}%")
                
        elif args.batch:
            results = predictor.predict_batch(args.input_path)
            print("\nBatch Prediction Results:")
            for result in results:
                print(f"\nFile: {result['filename']}")
                print(f"Prediction: {result['predicted_class']}")
                print(f"Confidence: {result['confidence']:.2f}%")
        else:
            result = predictor.predict_single(args.input_path)
            print("\nPrediction Results:")
            print(f"File: {result['filename']}")
            print(f"Prediction: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.2f}%")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())