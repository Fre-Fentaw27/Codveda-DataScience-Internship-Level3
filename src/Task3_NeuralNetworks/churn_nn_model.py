import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class ChurnNeuralNetwork:
    def __init__(self, input_dim):
        self.model = None
        self.input_dim = input_dim
        self.history = None
    
    def build_model(self, hidden_layers=[64, 32], dropout_rate=0.3, learning_rate=0.001):
        """Build the neural network architecture"""
        print("🏗️  BUILDING NEURAL NETWORK")
        print("="*60)
        
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(hidden_layers[0], activation='relu', input_shape=(self.input_dim,)))
        model.add(layers.Dropout(dropout_rate))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer (binary classification)
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("✅ Model architecture:")
        model.summary()
        
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the neural network"""
        print("\n🎯 TRAINING NEURAL NETWORK")
        print("="*60)
        
        # Early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        print("✅ Training completed!")
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("\n📊 MODEL EVALUATION")
        print("="*60)
        
        # Predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"📈 Test Metrics:")
        print(f"  Loss:      {test_loss:.4f}")
        print(f"  Accuracy:  {test_accuracy:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall:    {test_recall:.4f}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        report = classification_report(y_test, y_pred, target_names=['Not Churn', 'Churn'])
        print(report)
        
        return test_loss, test_accuracy, test_precision, test_recall, y_pred
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("❌ No training history available")
            return
        
        os.makedirs('../../outputs/Task3_NeuralNetworks/', exist_ok=True)
        
        # Plot accuracy
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('../../outputs/Task3_NeuralNetworks/training_history.png')
        plt.close()
        
        print("✅ Training history plot saved")
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Churn', 'Churn'],
                   yticklabels=['Not Churn', 'Churn'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        plt.savefig('../../outputs/Task3_NeuralNetworks/confusion_matrix.png')
        plt.close()
        
        print("✅ Confusion matrix saved")
    
    def save_model(self, filepath):
        """Save the trained model"""
        os.makedirs('../../models/Task3_NeuralNetworks/', exist_ok=True)
        self.model.save(filepath)
        print(f"✅ Model saved to: {filepath}")

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
    """Main function for neural network training"""
    print("🤖 STARTING NEURAL NETWORK TRAINING")
    print("="*60)
    
    # Load processed data
    processed_data = load_processed_data()
    if processed_data is None:
        return
    
    X_train, X_test, y_train, y_test, scaler = processed_data.values()
    
    # Further split training data for validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"📊 Data shapes:")
    print(f"  Training:   {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test:       {X_test.shape}")
    
    # Build and train model
    nn = ChurnNeuralNetwork(input_dim=X_train.shape[1])
    nn.build_model(hidden_layers=[64, 32, 16], dropout_rate=0.3, learning_rate=0.001)
    nn.train_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    
    # Evaluate model
    test_loss, test_accuracy, test_precision, test_recall, y_pred = nn.evaluate_model(X_test, y_test)
    
    # Create visualizations
    nn.plot_training_history()
    nn.plot_confusion_matrix(y_test, y_pred)
    
    # Save model
    nn.save_model('../../models/Task3_NeuralNetworks/best_churn_model.h5')
    
    # Save performance metrics
    with open('../../outputs/Task3_NeuralNetworks/model_performance.txt', 'w') as f:
        f.write("NEURAL NETWORK PERFORMANCE METRICS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Test Loss:      {test_loss:.4f}\n")
        f.write(f"Test Accuracy:  {test_accuracy:.4f}\n")
        f.write(f"Test Precision: {test_precision:.4f}\n")
        f.write(f"Test Recall:    {test_recall:.4f}\n")
    
    print("\n🎉 NEURAL NETWORK TRAINING COMPLETED!")

if __name__ == "__main__":
    main()