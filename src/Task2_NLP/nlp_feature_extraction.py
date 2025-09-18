import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

def extract_features():
    print("🔧 STARTING FEATURE EXTRACTION")
    print("="*60)
    
    # Load cleaned data
    try:
        df = pd.read_csv('../../data/processed/cleaned_sentiment_task2.csv')
        print(f"✅ Loaded cleaned data. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    except FileNotFoundError:
        print("❌ Cleaned data not found. Run preprocessing first.")
        return None, None, None, None
    
    # Detect text and label columns
    text_column = 'cleaned_text' if 'cleaned_text' in df.columns else df.columns[0]
    label_column = None
    
    for col in df.columns:
        if col != text_column and df[col].dtype == 'object' and df[col].nunique() < 10:
            label_column = col
            break
    
    if label_column is None:
        print("❌ Could not find label column")
        return None, None, None, None
    
    print(f"📝 Text column: {text_column}")
    print(f"🏷️  Label column: {label_column}")
    print(f"📊 Class distribution:\n{df[label_column].value_counts()}")
    
    # Extract features using TF-IDF
    print("\n🎯 EXTRACTING TF-IDF FEATURES")
    print("="*60)
    
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X = tfidf_vectorizer.fit_transform(df[text_column])
    y = df[label_column]
    
    print(f"✅ Feature matrix shape: {X.shape}")
    print(f"✅ Vocabulary size: {len(tfidf_vectorizer.get_feature_names_out())}")
    
    # Show top features
    feature_names = tfidf_vectorizer.get_feature_names_out()
    print(f"\n📋 Sample features: {feature_names[:10]}...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Train set: {X_train.shape}")
    print(f"📊 Test set: {X_test.shape}")
    
    # Save features and vectorizer
    os.makedirs('../../data/processed/features/', exist_ok=True)
    
    with open('../../data/processed/features/tfidf_features.pkl', 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)
    
    with open('../../data/processed/features/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    
    print("✅ Features saved to: ../../data/processed/features/")
    
    return X_train, X_test, y_train, y_test

def visualize_features(X, vectorizer, n_top_features=20):
    """Visualize top TF-IDF features"""
    print("\n📊 VISUALIZING TOP FEATURES")
    print("="*60)
    
    # Get feature names and importances
    feature_names = vectorizer.get_feature_names_out()
    feature_importances = np.asarray(X.mean(axis=0)).ravel()
    
    # Get top features
    top_indices = feature_importances.argsort()[-n_top_features:][::-1]
    top_features = feature_names[top_indices]
    top_importances = feature_importances[top_indices]
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_features)), top_importances)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('TF-IDF Importance')
    plt.title('Top 20 Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save visualization
    os.makedirs('../../outputs/Task2_NLP/', exist_ok=True)
    plt.savefig('../../outputs/Task2_NLP/feature_importance.png')
    plt.close()
    
    print("✅ Feature importance visualization saved")
    return top_features, top_importances

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = extract_features()
    
    if X_train is not None:
        # Load vectorizer for visualization
        with open('../../data/processed/features/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Visualize features
        visualize_features(X_train, vectorizer)
        
        print("\n🎉 FEATURE EXTRACTION COMPLETED!")
        print("="*60)