import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report

class ModelTrainer:
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Multinomial Naive Bayes': MultinomialNB(),
            'Support Vector Machine': SVC(kernel='linear', random_state=42, probability=True),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.results = {}
        self.best_model = None
        self.best_score = 0
    
    def load_data(self):
        """Load preprocessed features"""
        try:
            with open('../../data/processed/features/tfidf_features.pkl', 'rb') as f:
                X_train, X_test, y_train, y_test = pickle.load(f)
            
            print(f"✅ Loaded features. Train: {X_train.shape}, Test: {X_test.shape}")
            return X_train, X_test, y_train, y_test
            
        except FileNotFoundError:
            print("❌ Features not found. Run feature extraction first.")
            return None, None, None, None
    
    def train_models(self, X_train, y_train):
        """Train all models and compare performance"""
        print("\n🎯 TRAINING MODELS")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            self.results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            print(f"  {name}: CV Accuracy = {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
            # Update best model
            if cv_scores.mean() > self.best_score:
                self.best_score = cv_scores.mean()
                self.best_model = model
                self.best_model_name = name
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models on test set"""
        print("\n📊 MODEL EVALUATION")
        print("="*60)
        
        evaluation_results = {}
        
        for name, result in self.results.items():
            model = result['model']
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            evaluation_results[name] = {
                'test_accuracy': accuracy,
                'cv_accuracy': result['cv_mean']
            }
            
            print(f"{name}:")
            print(f"  CV Accuracy:    {result['cv_mean']:.4f}")
            print(f"  Test Accuracy:  {accuracy:.4f}")
            print()
        
        return evaluation_results
    
    def save_results(self, evaluation_results):
        """Save model results and best model"""
        # Save results to CSV
        results_df = pd.DataFrame(evaluation_results).T
        results_df.to_csv('../../models/Task2_NLP/model_comparison_results.csv')
        
        # Save best model
        os.makedirs('../../models/Task2_NLP/', exist_ok=True)
        with open('../../models/Task2_NLP/best_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Save model info
        model_info = {
            'best_model': self.best_model_name,
            'best_score': self.best_score,
            'all_results': evaluation_results
        }
        
        with open('../../models/Task2_NLP/model_info.pkl', 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"✅ Best model: {self.best_model_name} (Accuracy: {self.best_score:.4f})")
        print("✅ Results saved to: ../../models/Task2_NLP/")
    
    def visualize_results(self, evaluation_results):
        """Create visualization of model performance"""
        # Prepare data for plotting
        models = list(evaluation_results.keys())
        cv_scores = [evaluation_results[m]['cv_accuracy'] for m in models]
        test_scores = [evaluation_results[m]['test_accuracy'] for m in models]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, cv_scores, width, label='CV Accuracy', alpha=0.8)
        plt.bar(x + width/2, test_scores, width, label='Test Accuracy', alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.xticks(x, models, rotation=45, ha='right')
        plt.legend()
        plt.ylim(0, 1)
        plt.tight_layout()
        
        # Add value labels
        for i, (cv, test) in enumerate(zip(cv_scores, test_scores)):
            plt.text(i - width/2, cv + 0.01, f'{cv:.3f}', ha='center')
            plt.text(i + width/2, test + 0.01, f'{test:.3f}', ha='center')
        
        # Save plot
        os.makedirs('../../outputs/Task2_NLP/', exist_ok=True)
        plt.savefig('../../outputs/Task2_NLP/model_performance.png')
        plt.close()
        
        print("✅ Model performance visualization saved")

def main():
    print("🤖 STARTING MODEL TRAINING")
    print("="*60)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load data
    X_train, X_test, y_train, y_test = trainer.load_data()
    if X_train is None:
        return
    
    # Train models
    trainer.train_models(X_train, y_train)
    
    # Evaluate models
    evaluation_results = trainer.evaluate_models(X_test, y_test)
    
    # Save results
    trainer.save_results(evaluation_results)
    
    # Visualize results
    trainer.visualize_results(evaluation_results)
    
    print("\n🎉 MODEL TRAINING COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()