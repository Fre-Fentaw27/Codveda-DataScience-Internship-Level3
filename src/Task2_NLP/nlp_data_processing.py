import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import os

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class TextPreprocessor:
    def __init__(self, use_stemming=False):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer() if use_stemming else None
        self.lemmatizer = WordNetLemmatizer()
        self.use_stemming = use_stemming
        
    def clean_text(self, text):
        """
        Clean and preprocess a single text document
        """
        if pd.isna(text) or text == '':
            return ""
            
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming or lemmatization
        if self.use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        else:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
        # Join tokens back into a string
        return ' '.join(tokens)

def load_and_explore_data():
    """Load the raw data and explore its structure"""
    print("📂 Loading raw data...")
    
    try:
        # Load the dataset
        df = pd.read_csv('../../data/raw/Sentiment dataset.csv')
        print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
        
        # Display dataset information
        print("\n" + "="*60)
        print("📊 DATASET OVERVIEW")
        print("="*60)
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nData types:\n{df.dtypes}")
        print(f"\nMissing values:\n{df.isnull().sum()}")
        
        # Show first few rows
        print("\n" + "="*60)
        print("👀 FIRST 5 ROWS (RAW DATA)")
        print("="*60)
        print(df.head())
        
        return df
        
    except FileNotFoundError:
        print("❌ Error: File '../../data/raw/Sentiment dataset.csv' not found")
        print("Please make sure the file exists in the correct location")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def detect_text_column(df):
    """Automatically detect which column contains text data"""
    print("\n" + "="*60)
    print("🔍 DETECTING TEXT COLUMN")
    print("="*60)
    
    # Common text column names
    text_keywords = ['text', 'review', 'comment', 'message', 'content', 
                    'tweet', 'sentence', 'phrase', 'description', 'summary']
    
    # Check each column to find the most likely text column
    best_column = None
    max_text_length = 0
    
    for column in df.columns:
        # Skip obviously non-text columns
        if df[column].dtype == 'object' and df[column].nunique() > 5:
            avg_length = df[column].astype(str).apply(len).mean()
            
            print(f"  {column}: avg length = {avg_length:.1f} chars")
            
            # Check if column name contains text keywords
            column_lower = column.lower()
            is_likely_text = any(keyword in column_lower for keyword in text_keywords)
            
            if is_likely_text or avg_length > max_text_length:
                best_column = column
                max_text_length = avg_length
    
    if best_column:
        print(f"✅ Detected text column: '{best_column}' (avg length: {max_text_length:.1f} chars)")
        return best_column
    else:
        # If no obvious text column, use the first object column
        for column in df.columns:
            if df[column].dtype == 'object':
                print(f"⚠️  Using first object column: '{column}'")
                return column
        
        # Last resort: use first column
        print(f"⚠️  Using first column: '{df.columns[0]}'")
        return df.columns[0]

def detect_label_column(df, text_column):
    """Try to detect label column if it exists"""
    print("\n" + "="*60)
    print("🔍 DETECTING LABEL COLUMN")
    print("="*60)
    
    # Common label column names
    label_keywords = ['sentiment', 'label', 'category', 'class', 'rating', 
                     'emotion', 'score', 'target', 'type']
    
    for column in df.columns:
        if column != text_column:
            column_lower = column.lower()
            if any(keyword in column_lower for keyword in label_keywords):
                print(f"✅ Detected label column: '{column}'")
                return column
    
    # Check for columns with few unique values (likely labels)
    for column in df.columns:
        if column != text_column and df[column].nunique() < 20:
            print(f"⚠️  Possible label column (few unique values): '{column}'")
            print(f"    Unique values: {df[column].nunique()}")
            return column
    
    print("❌ No obvious label column detected")
    return None

def main():
    """Main function to process the data"""
    print("🔄 STARTING TEXT PREPROCESSING")
    print("="*60)
    
    # Load and explore raw data
    df = load_and_explore_data()
    if df is None:
        return
    
    # Detect text column
    text_column = detect_text_column(df)
    
    # Show sample of raw text
    print("\n" + "="*60)
    print("📝 SAMPLE RAW TEXT (BEFORE CLEANING)")
    print("="*60)
    for i in range(min(3, len(df))):
        print(f"Sample {i+1}:")
        print(f"  {df[text_column].iloc[i]}")
        print()
    
    # Detect label column
    label_column = detect_label_column(df, text_column)
    if label_column:
        print(f"\nLabel distribution:")
        print(df[label_column].value_counts())
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(use_stemming=False)
    
    # Clean the text data
    print("\n" + "="*60)
    print("🧹 CLEANING TEXT DATA...")
    print("="*60)
    
    df_clean = df.copy()
    df_clean['cleaned_text'] = df_clean[text_column].apply(preprocessor.clean_text)
    
    # Show sample of cleaned text
    print("\n" + "="*60)
    print("📝 SAMPLE CLEANED TEXT (AFTER CLEANING)")
    print("="*60)
    for i in range(min(3, len(df_clean))):
        print(f"Sample {i+1}:")
        print(f"  Original: {df[text_column].iloc[i][:100]}...")
        print(f"  Cleaned:  {df_clean['cleaned_text'].iloc[i]}")
        print()
    
    # Keep the label column if it exists
    if label_column:
        df_clean[label_column] = df[label_column]
    
    # Show cleaning statistics
    print("\n" + "="*60)
    print("📊 CLEANING STATISTICS")
    print("="*60)
    
    original_avg_length = df[text_column].astype(str).apply(len).mean()
    cleaned_avg_length = df_clean['cleaned_text'].apply(len).mean()
    reduction = ((original_avg_length - cleaned_avg_length) / original_avg_length) * 100
    
    print(f"Original average text length: {original_avg_length:.1f} characters")
    print(f"Cleaned average text length:  {cleaned_avg_length:.1f} characters")
    print(f"Length reduction:             {reduction:.1f}%")
    
    empty_count = (df_clean['cleaned_text'].str.strip() == '').sum()
    print(f"Empty texts after cleaning:   {empty_count}/{len(df_clean)}")
    
    # Save cleaned data
    print("\n" + "="*60)
    print("💾 SAVING CLEANED DATA")
    print("="*60)
    
    # Create directories if they don't exist
    os.makedirs('../../data/processed/', exist_ok=True)
    
    output_path = '../../data/processed/cleaned_sentiment_task2.csv'
    df_clean.to_csv(output_path, index=False)
    
    print(f"✅ Cleaned data saved to: {output_path}")
    print(f"✅ Shape of cleaned data: {df_clean.shape}")
    
    # Show final sample
    print("\n" + "="*60)
    print("👀 FINAL CLEANED DATA SAMPLE")
    print("="*60)
    if label_column:
        print(df_clean[['cleaned_text', label_column]].head())
    else:
        print(df_clean[['cleaned_text']].head())

if __name__ == "__main__":
    main()