import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterGrid
import pandas as pd

class HyperparameterTuner:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.results = []
        self.best_params = None
        self.best_score = 0
    
    def create_model(self, hidden_layers, dropout_rate, learning_rate):
        """Create model with given hyperparameters"""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(hidden_layers[0], activation='relu', input_shape=(self.input_dim,)))
        model.add(layers.Dropout(dropout_rate))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_and_evaluate(self, params, X_train, y_train, X_val, y_val):
        """Train and evaluate model with given parameters"""
        print(f"🔧 Testing parameters: {params}")
        
        model = self.create_model(
            params['hidden_layers'],
            params['dropout_rate'],
            params['learning_rate']
        )
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=0
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=params['batch_size'],
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate on validation set
        val_loss, val_accuracy, val_precision, val_recall = model.evaluate(X_val, y_val, verbose=0)
        
        # Store results
        result = {
            'params': params,
            'val_accuracy': val_accuracy,
            'val_loss': val_loss,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'epochs': len(history.history['loss']),
            'history': history.history
        }
        
        self.results.append(result)
        
        # Update best parameters
        if val_accuracy > self.best_score:
            self.best_score = val_accuracy
            self.best_params = params
            self.best_model = model
        
        print(f"   ✅ Val Accuracy: {val_accuracy:.4f}, Val Loss: {val_loss:.4f}")
        
        return result
    
    def grid_search(self, param_grid, X_train, y_train, X_val, y_val):
        """Perform grid search over parameter grid"""
        print("🎯 STARTING HYPERPARAMETER TUNING")
        print("="*60)
        
        # Generate all parameter combinations
        all_params = list(ParameterGrid(param_grid))
        print(f"📊 Testing {len(all_params)} parameter combinations...")
        
        for i, params in enumerate(all_params, 1):
            print(f"\n🔍 [{i}/{len(all_params)}]")
            self.train_and_evaluate(params, X_train, y_train, X_val, y_val)
        
        return self.results
    
    def get_best_model(self):
        """Get the best model from tuning"""
        return self.best_model, self.best_params
    
    def save_results(self):
        """Save tuning results to CSV"""
        os.makedirs('../../outputs/Task3_NeuralNetworks/', exist_ok=True)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame([{
            'hidden_layers': str(r['params']['hidden_layers']),
            'dropout_rate': r['params']['dropout_rate'],
            'learning_rate': r['params']['learning_rate'],
            'batch_size': r['params']['batch_size'],
            'val_accuracy': r['val_accuracy'],
            'val_loss': r['val_loss'],
            'val_precision': r['val_precision'],
            'val_recall': r['val_recall'],
            'epochs': r['epochs']
        } for r in self.results])
        
        # Sort by validation accuracy
        results_df = results_df.sort_values('val_accuracy', ascending=False)
        
        # Save to CSV
        results_df.to_csv('../../outputs/Task3_NeuralNetworks/hyperparameter_tuning_results.csv', index=False)
        
        print("✅ Tuning results saved to CSV")
        return results_df
    
    def plot_tuning_results(self):
        """Create visualizations of tuning results"""
        os.makedirs('../../outputs/Task3_NeuralNetworks/', exist_ok=True)
        
        if not self.results:
            print("❌ No results to plot")
            return
        
        # Prepare data for plotting
        results_df = self.save_results()
        
        # Plot 1: Accuracy vs Learning Rate
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        for dropout in results_df['dropout_rate'].unique():
            subset = results_df[results_df['dropout_rate'] == dropout]
            plt.scatter(subset['learning_rate'], subset['val_accuracy'], 
                       alpha=0.7, label=f'Dropout: {dropout}')
        plt.xscale('log')
        plt.xlabel('Learning Rate (log scale)')
        plt.ylabel('Validation Accuracy')
        plt.title('Accuracy vs Learning Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy vs Batch Size
        plt.subplot(1, 2, 2)
        for lr in results_df['learning_rate'].unique():
            subset = results_df[results_df['learning_rate'] == lr]
            plt.scatter(subset['batch_size'], subset['val_accuracy'], 
                       alpha=0.7, label=f'LR: {lr:.0e}')
        plt.xlabel('Batch Size')
        plt.ylabel('Validation Accuracy')
        plt.title('Accuracy vs Batch Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../../outputs/Task3_NeuralNetworks/hyperparameter_tuning_analysis.png')
        plt.close()
        
        print("✅ Tuning analysis plots saved")

def load_processed_data():
    """Load processed churn data"""
    try:
        with open('../../data/processed/churn_processed.pkl', 'rb') as f:
            processed_data = pickle.load(f)
        
        print("✅ Processed data loaded successfully!")
        return processed_data
        
    except FileNotFoundError:
        print("❌ Processed data not found. Run data processing first.")
        return None

def main():
    """Main function for hyperparameter tuning"""
    print("🎛️  STARTING HYPERPARAMETER TUNING")
    print("="*60)
    
    # Load processed data
    processed_data = load_processed_data()
    if processed_data is None:
        return
    
    X_train, X_test, y_train, y_test, scaler = processed_data.values()
    
    # Split training data for validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"📊 Data shapes:")
    print(f"  Training:   {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test:       {X_test.shape}")
    
    # Define parameter grid for tuning
    param_grid = {
        'hidden_layers': [
            [64, 32],
            [128, 64],
            [64, 32, 16],
            [128, 64, 32]
        ],
        'dropout_rate': [0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'batch_size': [16, 32, 64]
    }
    
    # Initialize tuner
    tuner = HyperparameterTuner(input_dim=X_train.shape[1])
    
    # Perform grid search
    results = tuner.grid_search(param_grid, X_train, y_train, X_val, y_val)
    
    # Get best model and parameters
    best_model, best_params = tuner.get_best_model()
    
    print(f"\n🎉 BEST PARAMETERS FOUND:")
    print(f"   Hidden Layers: {best_params['hidden_layers']}")
    print(f"   Dropout Rate:  {best_params['dropout_rate']}")
    print(f"   Learning Rate: {best_params['learning_rate']}")
    print(f"   Batch Size:    {best_params['batch_size']}")
    print(f"   Validation Accuracy: {tuner.best_score:.4f}")
    
    # Save results and visualizations
    tuner.save_results()
    tuner.plot_tuning_results()
    
    # Save best model
    os.makedirs('../../models/Task3_NeuralNetworks/', exist_ok=True)
    best_model.save('../../models/Task3_NeuralNetworks/tuned_churn_model.h5')
    print("✅ Best tuned model saved")
    
    # Evaluate best model on test set
    print(f"\n📊 EVALUATING BEST MODEL ON TEST SET:")
    test_loss, test_accuracy, test_precision, test_recall = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Accuracy:  {test_accuracy:.4f}")
    print(f"   Test Loss:      {test_loss:.4f}")
    print(f"   Test Precision: {test_precision:.4f}")
    print(f"   Test Recall:    {test_recall:.4f}")
    
    # Save final performance
    with open('../../outputs/Task3_NeuralNetworks/best_model_performance.txt', 'w') as f:
        f.write("BEST MODEL PERFORMANCE AFTER TUNING\n")
        f.write("="*50 + "\n\n")
        f.write(f"Best Parameters:\n")
        f.write(f"  Hidden Layers: {best_params['hidden_layers']}\n")
        f.write(f"  Dropout Rate:  {best_params['dropout_rate']}\n")
        f.write(f"  Learning Rate: {best_params['learning_rate']}\n")
        f.write(f"  Batch Size:    {best_params['batch_size']}\n\n")
        f.write(f"Test Accuracy:  {test_accuracy:.4f}\n")
        f.write(f"Test Loss:      {test_loss:.4f}\n")
        f.write(f"Test Precision: {test_precision:.4f}\n")
        f.write(f"Test Recall:    {test_recall:.4f}\n")
    
    print("\n🎉 HYPERPARAMETER TUNING COMPLETED!")

if __name__ == "__main__":
    main()