import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_churn_data():
    """Load and combine churn datasets"""
    print("📂 LOADING CHURN DATA")
    print("="*60)
    
    try:
        # Load both datasets
        train_df = pd.read_csv('../../data/raw/churn-bigml-80.csv')
        test_df = pd.read_csv('../../data/raw/churn-bigml-20.csv')
        
        # Combine datasets
        df = pd.concat([train_df, test_df], ignore_index=True)
        print(f"✅ Data loaded successfully! Shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print("❌ Churn data files not found.")
        print("Please download from: https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets")
        return None

def explore_churn_data(df):
    """Explore the churn dataset"""
    print("\n🔍 DATA EXPLORATION")
    print("="*60)
    
    print("📊 Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    print("\n📋 Data Types:")
    print(df.dtypes)
    
    print("\n🔍 Missing Values:")
    print(df.isnull().sum())
    
    print("\n📈 Target Distribution (Churn):")
    print(df['Churn'].value_counts())
    print(f"Churn Rate: {(df['Churn'].value_counts()[True] / len(df)) * 100:.2f}%")
    
    return df

def preprocess_churn_data(df):
    """Preprocess the churn data for neural network"""
    print("\n🧹 PREPROCESSING DATA")
    print("="*60)
    
    df_processed = df.copy()
    
    # Convert boolean target to binary
    df_processed['Churn'] = df_processed['Churn'].astype(int)
    
    # Handle categorical variables
    categorical_cols = ['State', 'International plan', 'Voice mail plan']
    
    print("🔤 Encoding categorical variables...")
    for col in categorical_cols:
        if col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                print(f"  Encoded {col}: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    # Separate features and target
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']
    
    # Handle numerical features - scale them
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print("📊 Scaling numerical features...")
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Data preprocessing completed!")
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    print(f"   Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, scaler

def visualize_data(df):
    """Create visualizations of the churn data"""
    print("\n📊 CREATING VISUALIZATIONS")
    print("="*60)
    
    os.makedirs('../../outputs/Task3_NeuralNetworks/', exist_ok=True)
    
    # Churn distribution
    plt.figure(figsize=(10, 6))
    df['Churn'].value_counts().plot(kind='bar')
    plt.title('Churn Distribution')
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.savefig('../../outputs/Task3_NeuralNetworks/churn_distribution.png')
    plt.close()
    
    # Correlation heatmap (for numerical features)
    numerical_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 10))
    sns.heatmap(numerical_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('../../outputs/Task3_NeuralNetworks/correlation_heatmap.png')
    plt.close()
    
    print("✅ Visualizations saved to outputs/Task3_NeuralNetworks/")

def main():
    """Main function for data processing"""
    print("🔄 STARTING CHURN DATA PROCESSING")
    print("="*60)
    
    # Load data
    df = load_churn_data()
    if df is None:
        return
    
    # Explore data
    explore_churn_data(df)
    
    # Create visualizations
    visualize_data(df)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_churn_data(df)
    
    # Save processed data
    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler
    }
    
    import pickle
    os.makedirs('../../data/processed/', exist_ok=True)
    with open('../../data/processed/churn_processed.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
    
    print("✅ Processed data saved to: ../../data/processed/churn_processed.pkl")
    print("\n🎉 DATA PROCESSING COMPLETED!")

if __name__ == "__main__":
    main()