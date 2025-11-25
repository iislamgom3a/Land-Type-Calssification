# ============================================================================
# File: evaluator.py
# Description: Model evaluation and metrics calculation
# ============================================================================

import numpy as np
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, classification_report,
    confusion_matrix
)


class ModelEvaluator:
    """Handles model evaluation and metrics calculation"""
    
    def __init__(self, model):
        self.model = model
        self.prediction_time = 0
        
    def evaluate(self, X_test, Y_test, batch_size=16):
        """
        Evaluate model on test set
        
        Args:
            X_test, Y_test: Test data
            batch_size: Batch size for evaluation
            
        Returns:
            tuple: (test_loss, test_accuracy)
        """
        print(f"\n{'='*60}")
        print("📊 Evaluating model on test set...")
        print(f"{'='*60}\n")
        
        test_loss, test_acc = self.model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
        print(f"\n✅ Test Loss: {test_loss:.4f}")
        print(f"✅ Test Accuracy: {test_acc:.4f}")
        return test_loss, test_acc
    
    def predict(self, X_test):
        """
        Make predictions on test set
        
        Args:
            X_test: Test features
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        print(f"\n{'='*60}")
        print("🔮 Making predictions...")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        predictions = self.model.predict(X_test, verbose=1)
        self.prediction_time = time.time() - start_time
        print(f"\n✅ Prediction completed in {self.prediction_time:.2f} seconds")
        return predictions
    
    def calculate_metrics(self, Y_test, pred_test):
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            Y_test: True labels (one-hot)
            pred_test: Predicted probabilities
            
        Returns:
            dict: Dictionary of metrics
        """
        y_pred_labels = np.argmax(pred_test, axis=1)
        y_true_labels = np.argmax(Y_test, axis=1)
        
        metrics = {
            'accuracy': accuracy_score(y_true_labels, y_pred_labels),
            'precision': precision_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0),
            'recall': recall_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0),
            'mcc': matthews_corrcoef(y_true_labels, y_pred_labels)
        }
        
        return metrics, y_true_labels, y_pred_labels
    
    def print_metrics(self, metrics, total_params, training_time, class_names=None):
        """
        Print all evaluation metrics in formatted manner
        
        Args:
            metrics: Dictionary of metrics
            total_params: Number of model parameters
            training_time: Training time in seconds
            class_names: List of class names (optional)
        """
        print("\n" + "="*60)
        print("📈 MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision:       {metrics['precision']:.4f}")
        print(f"Recall:          {metrics['recall']:.4f}")
        print(f"F1-Score:        {metrics['f1_score']:.4f}")
        print(f"MCC:             {metrics['mcc']:.4f}")
        print("-"*60)
        print(f"Model Complexity: {total_params:,} parameters")
        print(f"Training Time:    {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        print(f"Prediction Time:  {self.prediction_time:.2f} seconds")
        print("="*60)
    
    def print_classification_report(self, y_true, y_pred, class_names=None):
        """
        Print detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
        """
        print("\n" + "="*60)
        print("📋 DETAILED CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
        print("="*60)
    
    def get_confusion_matrix(self, y_true, y_pred):
        """
        Calculate confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            np.ndarray: Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)