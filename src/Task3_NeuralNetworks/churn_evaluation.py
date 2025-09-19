import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, f1_score,
                           roc_curve, auc, precision_recall_curve, RocCurveDisplay)
import pandas as pd

class ChurnModelEvaluator:
    def __init__(self):
        self.model = None
        self.X_test = None
        self.y_test = None
        self.scaler = None
    
    def load_model_and_data(self):
        """Load trained model and test data"""
        try:
            # Load processed data
            with open('../../data/processed/churn_processed.pkl', 'rb') as f:
                processed_data = pickle.load(f)
            
            self.X_test = processed_data['X_test']
            self.y_test = processed_data['y_test']
            self.scaler = processed_data['scaler']
            
            # Load best model (try tuned first, then regular)
            model_paths = [
                '../../models/Task3_NeuralNetworks/tuned_churn_model.h5',
                '../../models/Task3_NeuralNetworks/best_churn_model.h5'
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    self.model = keras.models.load_model(model_path)
                    print(f"✅ Model loaded from: {model_path}")
                    break
            
            if self.model is None:
                print("❌ No trained model found. Please train a model first.")
                return False
            
            print("✅ Data and model loaded successfully!")
            print(f"   Test data shape: {self.X_test.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading files: {e}")
            return False
    
    def comprehensive_evaluation(self):
        """Perform comprehensive model evaluation"""
        print("\n📈 COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        # Make predictions
        y_pred_proba = self.model.predict(self.X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            self.X_test, self.y_test, verbose=0
        )
        
        f1 = f1_score(self.y_test, y_pred)
        
        print("📊 PERFORMANCE METRICS:")
        print(f"   Accuracy:  {test_accuracy:.4f}")
        print(f"   Precision: {test_precision:.4f}")
        print(f"   Recall:    {test_recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   Loss:      {test_loss:.4f}")
        
        # Detailed classification report
        print(f"\n📋 CLASSIFICATION REPORT:")
        report = classification_report(self.y_test, y_pred, 
                                     target_names=['Not Churn', 'Churn'])
        print(report)
        
        return y_pred, y_pred_proba, {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': f1,
            'loss': test_loss
        }
    
    def plot_confusion_matrix(self, y_pred):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Churn', 'Churn'],
                   yticklabels=['Not Churn', 'Churn'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        os.makedirs('../../outputs/Task3_NeuralNetworks/', exist_ok=True)
        plt.savefig('../../outputs/Task3_NeuralNetworks/confusion_matrix.png')
        plt.close()
        
        print("✅ Confusion matrix saved")
        
        return cm
    
    def plot_roc_curve(self, y_pred_proba):
        """Plot ROC curve and calculate AUC"""
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.savefig('../../outputs/Task3_NeuralNetworks/roc_curve.png')
        plt.close()
        
        print(f"✅ ROC curve saved (AUC: {roc_auc:.3f})")
        
        return roc_auc
    
    def plot_precision_recall_curve(self, y_pred_proba):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'Precision-Recall curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        plt.savefig('../../outputs/Task3_NeuralNetworks/precision_recall_curve.png')
        plt.close()
        
        print(f"✅ Precision-Recall curve saved (AUC: {pr_auc:.3f})")
        
        return pr_auc
    
    def plot_prediction_distribution(self, y_pred_proba):
        """Plot distribution of prediction probabilities"""
        plt.figure(figsize=(10, 6))
        
        # Plot histogram of predictions
        plt.hist(y_pred_proba[self.y_test == 0], bins=30, alpha=0.7, 
                label='Not Churn', color='green')
        plt.hist(y_pred_proba[self.y_test == 1], bins=30, alpha=0.7, 
                label='Churn', color='red')
        
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Probabilities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('../../outputs/Task3_NeuralNetworks/prediction_distribution.png')
        plt.close()
        
        print("✅ Prediction distribution plot saved")
    
    def feature_importance_analysis(self):
        """Analyze feature importance (for neural networks)"""
        # This is a simplified approach for neural networks
        # For proper feature importance, consider using permutation importance
        
        print("\n🔍 FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Get the weights from the first layer
        first_layer_weights = self.model.layers[0].get_weights()[0]
        feature_importance = np.mean(np.abs(first_layer_weights), axis=1)
        
        # Load original data to get feature names
        try:
            df = pd.read_csv('../../data/raw/churn-bigml-80.csv')
            feature_names = df.drop('Churn', axis=1).columns.tolist()
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print("📊 Top 10 Most Important Features:")
            print(importance_df.head(10).to_string(index=False))
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            plt.barh(importance_df['feature'][:15], importance_df['importance'][:15])
            plt.xlabel('Importance (Average Absolute Weight)')
            plt.title('Top 15 Feature Importances')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            plt.savefig('../../outputs/Task3_NeuralNetworks/feature_importance.png')
            plt.close()
            
            print("✅ Feature importance plot saved")
            
            return importance_df
            
        except Exception as e:
            print(f"⚠️  Could not load feature names: {e}")
            return None
    
    def save_evaluation_results(self, metrics, cm, roc_auc, pr_auc):
        """Save all evaluation results to files"""
        os.makedirs('../../outputs/Task3_NeuralNetworks/', exist_ok=True)
        
        # Save metrics to text file
        with open('../../outputs/Task3_NeuralNetworks/comprehensive_evaluation.txt', 'w') as f:
            f.write("COMPREHENSIVE MODEL EVALUATION RESULTS\n")
            f.write("="*50 + "\n\n")
            
            f.write("📊 PERFORMANCE METRICS:\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1-Score:  {metrics['f1']:.4f}\n")
            f.write(f"Loss:      {metrics['loss']:.4f}\n")
            f.write(f"ROC AUC:   {roc_auc:.4f}\n")
            f.write(f"PR AUC:    {pr_auc:.4f}\n\n")
            
            f.write("📋 CONFUSION MATRIX:\n")
            f.write(f"True Negative: {cm[0,0]}\n")
            f.write(f"False Positive: {cm[0,1]}\n")
            f.write(f"False Negative: {cm[1,0]}\n")
            f.write(f"True Positive: {cm[1,1]}\n\n")
            
            f.write("📈 ADDITIONAL METRICS:\n")
            f.write(f"Churn Detection Rate: {(cm[1,1] / (cm[1,0] + cm[1,1])):.3f}\n")
            f.write(f"False Alarm Rate: {(cm[0,1] / (cm[0,0] + cm[0,1])):.3f}\n")
        
        print("✅ Comprehensive evaluation results saved")

def main():
    """Main function for model evaluation"""
    print("📊 STARTING COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    # Initialize evaluator
    evaluator = ChurnModelEvaluator()
    
    # Load model and data
    if not evaluator.load_model_and_data():
        return
    
    # Perform comprehensive evaluation
    y_pred, y_pred_proba, metrics = evaluator.comprehensive_evaluation()
    
    # Create visualizations
    cm = evaluator.plot_confusion_matrix(y_pred)
    roc_auc = evaluator.plot_roc_curve(y_pred_proba)
    pr_auc = evaluator.plot_precision_recall_curve(y_pred_proba)
    evaluator.plot_prediction_distribution(y_pred_proba)
    
    # Feature importance analysis
    importance_df = evaluator.feature_importance_analysis()
    
    # Save all results
    evaluator.save_evaluation_results(metrics, cm, roc_auc, pr_auc)
    
    print("\n🎉 COMPREHENSIVE EVALUATION COMPLETED!")
    print("="*60)
    print("📁 Check outputs in: ../../outputs/Task3_NeuralNetworks/")
    print("📊 All visualizations and results have been saved")

if __name__ == "__main__":
    main()