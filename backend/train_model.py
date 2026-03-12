import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATASET_PATH = os.path.join(os.path.dirname(__file__), 'spam_ham_dataset.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'spam_model.joblib')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'vectorizer.joblib')

def train():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    print(f"Loading dataset from {DATASET_PATH}...")
    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows:")
    print(df.head())
    
    # Use the correct columns from the CSV
    if 'label_num' in df.columns and 'text' in df.columns:
        # Use pre-labeled numeric column
        df_clean = df[['label_num', 'text']].copy()
        df_clean.columns = ['label', 'message']
        print("Using label_num and text columns")
    elif 'label' in df.columns and 'message' in df.columns:
        df_clean = df[['label', 'message']].copy()
    elif 'Category' in df.columns and 'Message' in df.columns:
        df_clean = df[['Category', 'Message']].copy()
        df_clean.columns = ['label', 'message']
    else:
        # Try first two columns if standard names not found
        df_clean = df.iloc[:, 1:3].copy()
        df_clean.columns = ['label', 'message']
    
    # Remove rows with missing values
    df_clean = df_clean.dropna()
    
    # Ensure labels are numeric (0 or 1)
    if df_clean['label'].dtype == 'object':
        print(f"Converting string labels to numeric...")
        df_clean['label'] = df_clean['label'].astype(str).str.lower().str.strip()
        df_clean['label'] = df_clean['label'].apply(
            lambda x: 1 if 'spam' in str(x).lower() else 0
        )
    
    # Ensure label is integer type
    df_clean['label'] = df_clean['label'].astype(int)
    
    print(f"\nLabel distribution:\n{df_clean['label'].value_counts()}")
    print(f"Unique classes: {sorted(df_clean['label'].unique())}")
    
    X = df_clean['message']
    y = df_clean['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Feature Extraction
    print("\nExtracting features using TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Feature matrix shape: {X_train_tfidf.shape}")
    print(f"Number of features: {len(vectorizer.get_feature_names_out())}")
    
    # Train model
    print("\nTraining Multinomial Naive Bayes model...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)'], zero_division=0))
    
    # Additional metrics
    print(f"\nPrediction distribution:")
    print(f"Ham predictions: {(y_pred == 0).sum()}")
    print(f"Spam predictions: {(y_pred == 1).sum()}")
    
    # Save model and vectorizer
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Vectorizer saved to {VECTORIZER_PATH}")

if __name__ == "__main__":
    train()
