"""
Batch Evaluation Script for Research Paper
============================================
Runs systematic experiments:
  - All models × all attacks × all explainers
  - Computes metrics (accuracy, precision, recall, F1, AUC)
  - Measures explanation fidelity and agreement
  - Exports structured results for paper tables/charts

Usage:
  python evaluate_batch.py [--samples N] [--output results.json]
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
import joblib

from adversarial_attacks import ATTACK_REGISTRY, apply_attack
from explainers import UnifiedExplainer


def load_models():
    """Load all available models."""
    models = {}
    base_dir = os.path.dirname(__file__)

    # TF-IDF
    tfidf_path = os.path.join(base_dir, 'spam_model.joblib')
    vec_path = os.path.join(base_dir, 'vectorizer.joblib')
    if os.path.exists(tfidf_path) and os.path.exists(vec_path):
        models['tfidf'] = {
            'model': joblib.load(tfidf_path),
            'vectorizer': joblib.load(vec_path),
            'name': 'TF-IDF + Naive Bayes',
        }

    # BERT
    bert_path = os.path.join(base_dir, 'bert_model.joblib')
    bert_cls_path = os.path.join(base_dir, 'spam_model_bert.joblib')
    if os.path.exists(bert_path) and os.path.exists(bert_cls_path):
        models['bert'] = {
            'model': joblib.load(bert_path),
            'classifier': joblib.load(bert_cls_path),
            'name': 'BERT + Logistic Regression',
        }

    return models


def make_predict_fn(model_key, models):
    """Create a predict_proba function for a given model."""
    if model_key == 'tfidf':
        vectorizer = models['tfidf']['vectorizer']
        model = models['tfidf']['model']
        def predict_fn(texts):
            X = vectorizer.transform(texts)
            return model.predict_proba(X)
        return predict_fn
    elif model_key == 'bert':
        bert = models['bert']['model']
        classifier = models['bert']['classifier']
        def predict_fn(texts):
            embeddings = bert.encode(texts)
            return classifier.predict_proba(embeddings)
        return predict_fn


def load_test_data(n_samples=None):
    """Load test dataset."""
    base_dir = os.path.dirname(__file__)
    
    # Try the main dataset
    for fname in ['spam_ham_dataset.csv', 'SMSSpamCollection']:
        path = os.path.join(base_dir, fname)
        if os.path.exists(path):
            if fname.endswith('.csv'):
                df = pd.read_csv(path)
            else:
                df = pd.read_csv(path, sep='\t', header=None, names=['label', 'message'])

            # Normalize columns
            if 'label_num' in df.columns and 'text' in df.columns:
                df = df[['label_num', 'text']].copy()
                df.columns = ['label', 'message']
            elif 'label' in df.columns and 'message' in df.columns:
                pass
            else:
                continue

            df = df.dropna()

            if df['label'].dtype == 'object':
                df['label'] = df['label'].apply(
                    lambda x: 1 if 'spam' in str(x).lower() else 0
                )
            df['label'] = df['label'].astype(int)

            # Use test split only
            _, test_df = train_test_split(df, test_size=0.2, random_state=42)

            if n_samples and n_samples < len(test_df):
                test_df = test_df.sample(n=n_samples, random_state=42)

            return test_df

    print("ERROR: No dataset found!")
    sys.exit(1)


def evaluate_clean(models, test_df):
    """Evaluate models on clean (unperturbed) test data."""
    results = {}

    for model_key in models:
        predict_fn = make_predict_fn(model_key, models)
        texts = test_df['message'].tolist()
        y_true = test_df['label'].values

        probas = predict_fn(texts)
        y_pred = np.argmax(probas, axis=1)
        y_score = probas[:, 1]

        results[model_key] = {
            'accuracy': round(accuracy_score(y_true, y_pred), 4),
            'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
            'recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
            'f1': round(f1_score(y_true, y_pred, zero_division=0), 4),
            'auc_roc': round(roc_auc_score(y_true, y_score), 4),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        }
        print(f"  {model_key}: Acc={results[model_key]['accuracy']}, F1={results[model_key]['f1']}, AUC={results[model_key]['auc_roc']}")

    return results


def evaluate_adversarial(models, test_df):
    """Evaluate model robustness under each adversarial attack."""
    results = {}

    # Only test on spam messages (adversarial attacks try to make spam look like ham)
    spam_df = test_df[test_df['label'] == 1]
    if len(spam_df) == 0:
        print("  WARNING: No spam samples in test set!")
        return results

    print(f"  Testing on {len(spam_df)} spam samples...")

    for attack_name, attack_info in ATTACK_REGISTRY.items():
        results[attack_name] = {
            'level': attack_info['level'],
            'description': attack_info['description'],
            'models': {},
        }

        # Generate adversarial versions
        perturbed_texts = [apply_attack(t, attack_name) for t in spam_df['message'].tolist()]
        y_true = spam_df['label'].values  # All should be 1 (spam)

        for model_key in models:
            predict_fn = make_predict_fn(model_key, models)

            # Original predictions
            orig_probas = predict_fn(spam_df['message'].tolist())
            orig_preds = np.argmax(orig_probas, axis=1)

            # Adversarial predictions
            adv_probas = predict_fn(perturbed_texts)
            adv_preds = np.argmax(adv_probas, axis=1)

            # How many predictions flipped (spam → ham)?
            flipped = int(np.sum(orig_preds != adv_preds))
            evasion_rate = round(flipped / len(spam_df), 4)

            # Average confidence drop
            orig_spam_conf = orig_probas[:, 1].mean()
            adv_spam_conf = adv_probas[:, 1].mean()
            confidence_drop = round(float(orig_spam_conf - adv_spam_conf), 4)

            results[attack_name]['models'][model_key] = {
                'evasion_rate': evasion_rate,
                'flipped_count': flipped,
                'total_samples': len(spam_df),
                'avg_orig_confidence': round(float(orig_spam_conf), 4),
                'avg_adv_confidence': round(float(adv_spam_conf), 4),
                'confidence_drop': confidence_drop,
            }

        print(f"  {attack_name}: " + " | ".join(
            f"{mk}={results[attack_name]['models'][mk]['evasion_rate']:.1%} evasion"
            for mk in models
        ))

    return results


def evaluate_explainability(models, test_df, n_explain=20):
    """Compare LIME vs SHAP explanations and measure fidelity."""
    results = {}
    explainer = UnifiedExplainer()

    # Sample a balanced subset for explanation analysis
    sample = test_df.groupby('label').apply(
        lambda x: x.sample(min(n_explain // 2, len(x)), random_state=42)
    ).reset_index(drop=True)

    print(f"  Explaining {len(sample)} samples per model...")

    for model_key in models:
        predict_fn = make_predict_fn(model_key, models)
        model_results = {
            'agreement_scores': [],
            'fidelity_scores_lime': [],
            'fidelity_scores_shap': [],
            'prediction_flip_rate_lime': 0,
            'prediction_flip_rate_shap': 0,
            'sample_explanations': [],
        }

        lime_flips = 0
        shap_flips = 0

        for idx, row in sample.iterrows():
            text = row['message']
            try:
                explanations = explainer.explain_all(text, predict_fn, num_features=5)

                # Agreement
                if explanations['agreement']['agreement_score'] is not None:
                    model_results['agreement_scores'].append(
                        explanations['agreement']['agreement_score']
                    )

                # Fidelity for LIME
                lime_fidelity = explainer.compute_explanation_fidelity(
                    text, predict_fn, explanations['lime'], top_k=3
                )
                model_results['fidelity_scores_lime'].append(lime_fidelity['fidelity_score'])
                if lime_fidelity['prediction_flipped']:
                    lime_flips += 1

                # Fidelity for SHAP
                if explanations['shap'].get('features'):
                    shap_fidelity = explainer.compute_explanation_fidelity(
                        text, predict_fn, explanations['shap'], top_k=3
                    )
                    model_results['fidelity_scores_shap'].append(shap_fidelity['fidelity_score'])
                    if shap_fidelity['prediction_flipped']:
                        shap_flips += 1

                # Store sample (first 5 only)
                if len(model_results['sample_explanations']) < 5:
                    model_results['sample_explanations'].append({
                        'text': text[:100],
                        'true_label': int(row['label']),
                        'lime_top3': explanations['lime']['top_spam_words'][:3],
                        'shap_top3': explanations['shap'].get('top_spam_words', [])[:3],
                        'agreement': explanations['agreement']['agreement_score'],
                    })

            except Exception as e:
                print(f"    Skipping sample due to error: {e}")
                continue

        n = len(sample)
        model_results['avg_agreement'] = round(np.mean(model_results['agreement_scores']), 4) if model_results['agreement_scores'] else None
        model_results['avg_fidelity_lime'] = round(np.mean(model_results['fidelity_scores_lime']), 4) if model_results['fidelity_scores_lime'] else None
        model_results['avg_fidelity_shap'] = round(np.mean(model_results['fidelity_scores_shap']), 4) if model_results['fidelity_scores_shap'] else None
        model_results['prediction_flip_rate_lime'] = round(lime_flips / max(n, 1), 4)
        model_results['prediction_flip_rate_shap'] = round(shap_flips / max(n, 1), 4)

        # Clean up large lists
        del model_results['agreement_scores']
        del model_results['fidelity_scores_lime']
        del model_results['fidelity_scores_shap']

        results[model_key] = model_results
        print(f"  {model_key}: LIME fidelity={model_results['avg_fidelity_lime']}, SHAP fidelity={model_results['avg_fidelity_shap']}, agreement={model_results['avg_agreement']}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Batch evaluation for spam detection research')
    parser.add_argument('--samples', type=int, default=200, help='Number of test samples')
    parser.add_argument('--explain-samples', type=int, default=20, help='Number of samples for explanation analysis')
    parser.add_argument('--output', type=str, default='experiment_results.json', help='Output file')
    args = parser.parse_args()

    print("=" * 60)
    print("SPAM DETECTION RESEARCH — BATCH EVALUATION")
    print("=" * 60)

    # Load
    print("\n[1/4] Loading models...")
    models = load_models()
    print(f"  Loaded: {list(models.keys())}")

    print(f"\n[2/4] Loading test data ({args.samples} samples)...")
    test_df = load_test_data(n_samples=args.samples)
    print(f"  Test set: {len(test_df)} samples (spam={sum(test_df['label']==1)}, ham={sum(test_df['label']==0)})")

    all_results = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_test_samples': len(test_df),
            'models': list(models.keys()),
            'attacks': list(ATTACK_REGISTRY.keys()),
        }
    }

    # Clean evaluation
    print("\n[3/4] Evaluating on clean data...")
    all_results['clean_metrics'] = evaluate_clean(models, test_df)

    # Adversarial evaluation
    print("\n[4/4] Evaluating adversarial robustness...")
    all_results['adversarial_metrics'] = evaluate_adversarial(models, test_df)

    # Explainability evaluation
    print(f"\n[BONUS] Explainability analysis ({args.explain_samples} samples)...")
    all_results['explainability_metrics'] = evaluate_explainability(
        models, test_df, n_explain=args.explain_samples
    )

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 60}")

    # Quick summary table
    print("\n--- QUICK SUMMARY ---")
    print(f"{'Model':<10} {'Accuracy':<10} {'F1':<10} {'AUC':<10}")
    for mk, metrics in all_results['clean_metrics'].items():
        print(f"{mk:<10} {metrics['accuracy']:<10} {metrics['f1']:<10} {metrics['auc_roc']:<10}")

    print(f"\n{'Attack':<18} {'Level':<12}", end="")
    for mk in models:
        print(f" {mk+' evasion':<16}", end="")
    print()
    for atk, info in all_results['adversarial_metrics'].items():
        print(f"{atk:<18} {info['level']:<12}", end="")
        for mk in models:
            rate = info['models'].get(mk, {}).get('evasion_rate', 'N/A')
            print(f" {rate:<16}", end="")
        print()


if __name__ == '__main__':
    main()
