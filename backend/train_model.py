import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH = os.path.join(os.path.dirname(__file__), 'spam_ham_dataset.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'spam_model.joblib')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'vectorizer.joblib')

def generate_visualizations(y_test, y_pred, y_pred_proba, model, output_dir):
    """Generate evaluation visualizations."""
    reports_dir = os.path.join(output_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - TF-IDF + Naive Bayes')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = os.path.join(reports_dir, 'confusion_matrix_tfidf.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved: {cm_path}")
    
    # 2. ROC Curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - TF-IDF + Naive Bayes')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    roc_path = os.path.join(reports_dir, 'roc_curve_tfidf.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC curve saved: {roc_path}")
    
    # 3. Metrics Summary Visualization
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    scores = [accuracy, precision, recall, f1, roc_auc]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'], alpha=0.8, edgecolor='black')
    plt.ylabel('Score')
    plt.title('Model Performance Metrics - TF-IDF + Naive Bayes')
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    metrics_path = os.path.join(reports_dir, 'metrics_summary_tfidf.png')
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Metrics summary saved: {metrics_path}")

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
    
    # Calculate AUC-ROC
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC-ROC Score: {auc:.4f}")
    
    # Generate visualizations
    generate_visualizations(y_test, y_pred, y_pred_proba, model, os.path.dirname(__file__))
    
    # Save model and vectorizer
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Vectorizer saved to {VECTORIZER_PATH}")

if __name__ == "__main__":
    train()
