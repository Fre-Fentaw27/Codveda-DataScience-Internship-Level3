import pickle
import pandas as pd
import numpy as np
import os
from nlp_data_processing import TextPreprocessor

class SentimentPredictor:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.preprocessor = TextPreprocessor(use_stemming=False)
        self.load_models()
    
    def load_models(self):
        """Load trained model and vectorizer"""
        try:
            # Load model
            with open('../../models/Task2_NLP/best_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            # Load vectorizer
            with open('../../data/processed/features/tfidf_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            print("✅ Models loaded successfully!")
            
        except FileNotFoundError:
            print("❌ Model files not found. Please train the model first.")
            self.model = None
            self.vectorizer = None
    
    def predict(self, text):
        """Predict sentiment for a single text"""
        if self.model is None or self.vectorizer is None:
            print("❌ Models not loaded. Cannot make predictions.")
            return None, None
        
        # Preprocess text
        cleaned_text = self.preprocessor.clean_text(text)
        
        # Vectorize text
        text_vector = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.model.predict(text_vector)[0]
        probability = self.model.predict_proba(text_vector)[0] if hasattr(self.model, 'predict_proba') else None
        
        return prediction, probability, cleaned_text
    
    def predict_batch(self, texts):
        """Predict sentiment for multiple texts"""
        if self.model is None or self.vectorizer is None:
            print("❌ Models not loaded. Cannot make predictions.")
            return None, None
        
        # Preprocess texts
        cleaned_texts = [self.preprocessor.clean_text(text) for text in texts]
        
        # Vectorize texts
        text_vectors = self.vectorizer.transform(cleaned_texts)
        
        # Predict
        predictions = self.model.predict(text_vectors)
        probabilities = self.model.predict_proba(text_vectors) if hasattr(self.model, 'predict_proba') else None
        
        return predictions, probabilities, cleaned_texts

def main():
    print("🔮 SENTIMENT PREDICTION")
    print("="*60)
    
    # Initialize predictor
    predictor = SentimentPredictor()
    
    if predictor.model is None:
        return
    
    # Sample texts for prediction
    sample_texts = [
        "I absolutely love this beautiful sunny day at the park!",
        "The traffic this morning was terrible and made me late for work.",
        "This product is okay, nothing special but it works fine.",
        "The customer service was absolutely horrible and rude.",
        "I'm so excited about my upcoming vacation to the beach!"
    ]
    
    print("📝 Sample Predictions:")
    print("="*60)
    
    for i, text in enumerate(sample_texts, 1):
        prediction, probability, cleaned_text = predictor.predict(text)
        
        print(f"\n📋 Text {i}:")
        print(f"   Original: {text}")
        print(f"   Cleaned:  {cleaned_text}")
        print(f"   Prediction: {prediction}")
        
        if probability is not None:
            # Get class names from model
            if hasattr(predictor.model, 'classes_'):
                classes = predictor.model.classes_
                prob_dict = {cls: f"{prob*100:.1f}%" for cls, prob in zip(classes, probability)}
                print(f"   Probabilities: {prob_dict}")
        
        print("-" * 50)
    
    # Batch prediction example
    print("\n🎯 Batch Prediction Results:")
    print("="*60)
    
    predictions, probabilities, cleaned_texts = predictor.predict_batch(sample_texts)
    
    if predictions is not None:
        results_df = pd.DataFrame({
            'Original_Text': sample_texts,
            'Cleaned_Text': cleaned_texts,
            'Prediction': predictions
        })
        
        print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()