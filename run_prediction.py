from predict import MelanomaPredictor

# Change these paths to your image and folder paths
SINGLE_IMAGE_PATH = "C:/Users/marti/Desktop/ICS 4.2/Project/archive/test/Malignant/5616.jpg"    # Change this to your image path
BATCH_FOLDER_PATH = "C:/Users/marti/Desktop/ICS 4.2/Project/archive/test/Malignant"         # Change this to your folder path
TEST_DIRECTORY_PATH = "C:/Users/marti/Desktop/ICS 4.2/Project/archive/test"    # Change this to your test directory containing benign and malignant folders
MODEL_PATH = "best_model.h5"                        # Change this if your model has a different name

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
    """Run evaluation on test directory with confusion matrix"""
    print("\n=== Full Evaluation with Confusion Matrix ===")
    try:
        print(f"\nEvaluating test directory: {TEST_DIRECTORY_PATH}")
        print("This directory should contain 'benign' and 'malignant' subdirectories")
        
        results, class_report, conf_matrix = predictor.evaluate_test_directory(TEST_DIRECTORY_PATH)
        
        print("\n--- Individual Predictions ---")
        for result in results:
            print(f"\nFile: {result['filename']}")
            print(f"Predicted class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            
        print("\nConfusion matrix has been saved as 'confusion_matrix.png'")
        
        # Print some key metrics from the classification report
        print("\n--- Key Metrics ---")
        print(f"Overall Accuracy: {class_report['accuracy']:.4f}")
        print("\nPer-Class Metrics:")
        for class_name in ['benign', 'malignant']:
            print(f"\n{class_name.capitalize()}:")
            print(f"Precision: {class_report[class_name]['precision']:.4f}")
            print(f"Recall: {class_report[class_name]['recall']:.4f}")
            print(f"F1-Score: {class_report[class_name]['f1-score']:.4f}")
            
    except Exception as e:
        print(f"Error with evaluation: {str(e)}")

def main():
    # Initialize the predictor
    predictor = MelanomaPredictor(model_path=MODEL_PATH)
    
    # 1. Run single image prediction
    run_single_prediction(predictor)
    
    # 2. Run batch prediction
    run_batch_prediction(predictor)
    
    # 3. Run full evaluation with confusion matrix
    run_evaluation(predictor)

if __name__ == "__main__":
    main()