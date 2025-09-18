import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, f1_score,
                           roc_curve, auc, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize

class ModelEvaluator:
    def __init__(self):
        self.best_model = None
        self.vectorizer = None
        self.X_test = None
        self.y_test = None
    
    def load_data_and_model(self):
        """Load test data and trained model"""
        try:
            # Load features
            with open('../../data/processed/features/tfidf_features.pkl', 'rb') as f:
                _, X_test, _, y_test = pickle.load(f)
            
            # Load vectorizer
            with open('../../data/processed/features/tfidf_vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            
            # Load best model
            with open('../../models/Task2_NLP/best_model.pkl', 'rb') as f:
                best_model = pickle.load(f)
            
            print("✅ Successfully loaded model and test data")
            return best_model, vectorizer, X_test, y_test
            
        except FileNotFoundError as e:
            print(f"❌ Error loading files: {e}")
            return None, None, None, None
    
    def evaluate_model(self, model, X_test, y_test):
        """Comprehensive model evaluation"""
        print("\n📈 COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"📊 Overall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Detailed classification report
        print(f"\n📋 Detailed Classification Report:")
        report = classification_report(y_test, y_pred)
        print(report)
        
        # Save report
        os.makedirs('../../outputs/Task2_NLP/', exist_ok=True)
        with open('../../outputs/Task2_NLP/classification_report.txt', 'w') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1-Score: {f1:.4f}\n\n")
            f.write(report)
        
        return y_pred, y_prob, accuracy, precision, recall, f1
    
    def plot_confusion_matrix(self, y_test, y_pred, class_names):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        plt.savefig('../../outputs/Task2_NLP/confusion_matrix.png')
        plt.close()
        
        print("✅ Confusion matrix saved")
    
    def plot_roc_curve(self, y_test, y_prob, class_names):
        """Plot ROC curve for multiclass classification"""
        if y_prob is None:
            print("⚠️  Model doesn't support probability predictions, skipping ROC curve")
            return
        
        # Binarize the output
        y_test_bin = label_binarize(y_test, classes=class_names)
        n_classes = len(class_names)
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot all ROC curves
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(class_names[i], roc_auc[i]))
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Multiclass Classification')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        plt.savefig('../../outputs/Task2_NLP/roc_curve.png')
        plt.close()
        
        print("✅ ROC curve saved")
    
    def save_final_results(self, accuracy, precision, recall, f1):
        """Save final results summary"""
        results_summary = f"""
RESULTS SUMMARY
===============

Overall Performance:
- Accuracy:  {accuracy:.4f}
- Precision: {precision:.4f}
- Recall:    {recall:.4f}
- F1-Score:  {f1:.4f}

Model Evaluation Completed Successfully!
"""
        
        with open('../../outputs/Task2_NLP/results_summary.txt', 'w') as f:
            f.write(results_summary)
        
        print("✅ Results summary saved")

def main():
    print("📊 STARTING MODEL EVALUATION")
    print("="*60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load data and model
    best_model, vectorizer, X_test, y_test = evaluator.load_data_and_model()
    if best_model is None:
        return
    
    # Get class names
    class_names = sorted(y_test.unique())
    print(f"📋 Classes: {class_names}")
    
    # Evaluate model
    y_pred, y_prob, accuracy, precision, recall, f1 = evaluator.evaluate_model(
        best_model, X_test, y_test
    )
    
    # Create visualizations
    evaluator.plot_confusion_matrix(y_test, y_pred, class_names)
    evaluator.plot_roc_curve(y_test, y_prob, class_names)
    
    # Save final results
    evaluator.save_final_results(accuracy, precision, recall, f1)
    
    print("\n🎉 MODEL EVALUATION COMPLETED!")
    print("="*60)
    print("📁 Check outputs in: ../../outputs/Task2_NLP/")

if __name__ == "__main__":
    main()