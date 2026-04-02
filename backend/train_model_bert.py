import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
import joblib

DATASET_PATH = os.path.join(os.path.dirname(__file__), 'spam_ham_dataset.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'spam_model_bert.joblib')
BERT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'bert_model.joblib')

def train():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Prepare data
    if 'label_num' in df.columns and 'text' in df.columns:
        df_clean = df[['label_num', 'text']].copy()
        df_clean.columns = ['label', 'message']
        print("Using label_num and text columns")
    
    df_clean = df_clean.dropna()
    
    # Convert to numeric labels
    if df_clean['label'].dtype == 'object':
        print(f"Converting string labels to numeric...")
        df_clean['label'] = df_clean['label'].apply(
            lambda x: 1 if 'spam' in str(x).lower() else 0
        )
    
    df_clean['label'] = df_clean['label'].astype(int)
    
    print(f"\nLabel distribution:\n{df_clean['label'].value_counts()}")
    
    X = df_clean['message']
    y = df_clean['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Load BERT model
    print("\nLoading BERT model (sentence-transformers)...")
    print("This may take a moment on first run...")
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get BERT embeddings
    print("\nGenerating BERT embeddings for training data...")
    X_train_embeddings = bert_model.encode(X_train.tolist(), show_progress_bar=True)
    
    print("Generating BERT embeddings for test data...")
    X_test_embeddings = bert_model.encode(X_test.tolist(), show_progress_bar=True)
    
    print(f"Embedding shape: {X_train_embeddings.shape}")
    
    # Train classifier on BERT embeddings
    print("\nTraining Logistic Regression on BERT embeddings...")
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train_embeddings, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test_embeddings)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"Model Accuracy (BERT): {accuracy:.4f}")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)'], zero_division=0))
    
    # Save models
    joblib.dump(bert_model, BERT_MODEL_PATH)
    joblib.dump(classifier, MODEL_PATH)
    print(f"\nBERT model saved to {BERT_MODEL_PATH}")
    print(f"Classifier saved to {MODEL_PATH}")
    print("\nYou can now use both TF-IDF and BERT models in the API!")

if __name__ == "__main__":
    train()
