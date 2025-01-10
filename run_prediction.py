from predict import MelanomaPredictor

# Path configurations
SINGLE_IMAGE_PATH = "C:/Users/marti/Desktop/ICS 4.2/Project/archive/test/Malignant/5616.jpg"
BATCH_FOLDER_PATH = "C:/Users/marti/Desktop/ICS 4.2/Project/archive/test/Malignant"
TEST_DIRECTORY_PATH = "C:/Users/marti/Desktop/ICS 4.2/Project/archive/test"
MODEL_PATH = "best_model.h5"

def run_single_prediction(predictor):
    """Run prediction on a single image"""
    print("\n=== Single Image Prediction ===")
    try:
        result = predictor.predict_single(SINGLE_IMAGE_PATH)
        print(f"\nResults for: {result['filename']}")
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2f}%")
    except Exception as e:
        print(f"Error with single image prediction: {str(e)}")

def run_batch_prediction(predictor):
    """Run prediction on a batch of images"""
    print("\n=== Batch Prediction ===")
    try:
        results = predictor.predict_batch(BATCH_FOLDER_PATH)
        for result in results:
            print(f"\nResults for: {result['filename']}")
            print(f"Predicted class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.2f}%")
    except Exception as e:
        print(f"Error with batch prediction: {str(e)}")

def run_evaluation(predictor):
    """Run evaluation on test directory with confusion matrix and detailed metrics"""
    print("\n=== Full Evaluation with Metrics ===")
    try:
        print(f"\nEvaluating test directory: {TEST_DIRECTORY_PATH}")
        print("This directory should contain 'benign' and 'malignant' subdirectories")
        
        results, conf_matrix, metrics = predictor.evaluate_test_directory(TEST_DIRECTORY_PATH)
        
        print("\n--- Individual Predictions ---")
        for result in results:
            print(f"\nFile: {result['filename']}")
            print(f"Predicted class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            
        print("\nConfusion matrix and ROC curve have been saved as 'confusion_matrix.png' and 'roc_curve.png'")
        
        # Combined Metrics
        print("\n=== Combined Model Metrics ===")
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Overall Sensitivity/Recall: {metrics['sensitivity']:.4f}")
        print(f"Overall Specificity: {metrics['specificity']:.4f}")
        print(f"Overall Precision: {metrics['precision']:.4f}")
        print(f"Overall F1 Score: {metrics['f1_score']:.4f}")
        print(f"AUC-ROC Score: {metrics['auc_roc']:.4f}")
        
        # Confusion Matrix Values
        tn, fp, fn, tp = conf_matrix.ravel()
        print("\n=== Confusion Matrix Breakdown ===")
        print(f"True Negatives (Correct Benign): {tn}")
        print(f"False Positives (Incorrect Malignant): {fp}")
        print(f"False Negatives (Incorrect Benign): {fn}")
        print(f"True Positives (Correct Malignant): {tp}")
        
        # Per-Class Metrics
        print("\n=== Per-Class Metrics ===")
        
        # Benign metrics
        print("\nBenign Class:")
        benign_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
        benign_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
        benign_f1 = 2 * (benign_precision * benign_recall) / (benign_precision + benign_recall) if (benign_precision + benign_recall) > 0 else 0
        print(f"Precision: {benign_precision:.4f}")
        print(f"Recall: {benign_recall:.4f}")
        print(f"F1-Score: {benign_f1:.4f}")
        
        # Malignant metrics
        print("\nMalignant Class:")
        malignant_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        malignant_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        malignant_f1 = 2 * (malignant_precision * malignant_recall) / (malignant_precision + malignant_recall) if (malignant_precision + malignant_recall) > 0 else 0
        print(f"Precision: {malignant_precision:.4f}")
        print(f"Recall: {malignant_recall:.4f}")
        print(f"F1-Score: {malignant_f1:.4f}")
            
    except Exception as e:
        print(f"Error with evaluation: {str(e)}")

def main():
    predictor = MelanomaPredictor(model_path=MODEL_PATH)
    
    run_single_prediction(predictor)
    run_batch_prediction(predictor)
    run_evaluation(predictor)

if __name__ == "__main__":
    main()