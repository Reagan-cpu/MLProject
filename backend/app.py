"""
SpamDetect API — Research-Grade Backend
========================================
Flask API integrating:
  - Dual-model inference (TF-IDF vs BERT)
  - Multi-method explainability (LIME + SHAP)
  - Multi-strategy adversarial robustness testing
  - Explanation fidelity measurement
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

from adversarial_attacks import ATTACK_REGISTRY, apply_attack, apply_all_attacks
from explainers import UnifiedExplainer

app = Flask(__name__)
CORS(app)

# ─── LOAD MODELS ──────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(__file__)

# TF-IDF Model
tfidf_model = None
vectorizer = None
TFIDF_MODEL_PATH = os.path.join(BASE_DIR, 'spam_model.joblib')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.joblib')

if os.path.exists(TFIDF_MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    tfidf_model = joblib.load(TFIDF_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

# BERT Model
bert_model = None
bert_classifier = None
BERT_MODEL_PATH = os.path.join(BASE_DIR, 'bert_model.joblib')
BERT_CLASSIFIER_PATH = os.path.join(BASE_DIR, 'spam_model_bert.joblib')

if os.path.exists(BERT_MODEL_PATH) and os.path.exists(BERT_CLASSIFIER_PATH):
    bert_model = joblib.load(BERT_MODEL_PATH)
    bert_classifier = joblib.load(BERT_CLASSIFIER_PATH)

# ─── EXPLAINER ────────────────────────────────────────────────────────────────

explainer = UnifiedExplainer(class_names=["Not Spam", "Spam"])


# ─── PREDICT FUNCTIONS ────────────────────────────────────────────────────────

def predict_proba_tfidf(texts):
    texts_tfidf = vectorizer.transform(texts)
    return tfidf_model.predict_proba(texts_tfidf)


def predict_proba_bert(texts):
    embeddings = bert_model.encode(texts)
    return bert_classifier.predict_proba(embeddings)


def get_predict_fn(model_type):
    if model_type == 'bert':
        return predict_proba_bert
    return predict_proba_tfidf


# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():
    """Classify a message and return prediction + LIME explanation."""
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    model_type = data.get('model', 'bert' if bert_model else 'tfidf')
    message = data['message']

    if model_type == 'bert':
        if bert_model is None or bert_classifier is None:
            return jsonify({'error': 'BERT model not trained. Run train_model_bert.py first.'}), 500
        predict_fn = predict_proba_bert
    elif model_type == 'tfidf':
        if tfidf_model is None or vectorizer is None:
            return jsonify({'error': 'TF-IDF model not trained. Run train_model.py first.'}), 500
        predict_fn = predict_proba_tfidf
    else:
        return jsonify({'error': 'Invalid model type. Use "bert" or "tfidf".'}), 400

    probabilities = predict_fn([message])[0]
    prediction = int(probabilities.argmax())
    label = "Spam" if prediction == 1 else "Not Spam"
    probability = float(probabilities[prediction])

    # LIME explanation
    lime_exp = explainer.explain_lime(message, predict_fn, num_features=6, num_samples=100)

    return jsonify({
        'prediction': label,
        'probability': round(probability * 100, 2),
        'is_spam': bool(prediction == 1),
        'model_used': model_type,
        'explanation': lime_exp['features'],
        'top_spam_words': lime_exp['top_spam_words'],
        'top_ham_words': lime_exp['top_ham_words'],
    })


@app.route('/explain', methods=['POST'])
def explain():
    """
    Full explainability comparison endpoint.
    Returns LIME, SHAP, agreement analysis, and fidelity scores.
    """
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    model_type = data.get('model', 'bert' if bert_model else 'tfidf')
    message = data['message']
    predict_fn = get_predict_fn(model_type)

    # Run all explainers
    explanations = explainer.explain_all(message, predict_fn, num_features=8)

    # Fidelity analysis
    lime_fidelity = explainer.compute_explanation_fidelity(
        message, predict_fn, explanations['lime'], top_k=3
    )
    shap_fidelity = None
    if explanations['shap'].get('features'):
        shap_fidelity = explainer.compute_explanation_fidelity(
            message, predict_fn, explanations['shap'], top_k=3
        )

    # Prediction
    probas = predict_fn([message])[0]
    prediction = int(probas.argmax())

    return jsonify({
        'message': message,
        'model_used': model_type,
        'prediction': "Spam" if prediction == 1 else "Not Spam",
        'probability': round(float(probas[prediction]) * 100, 2),
        'is_spam': bool(prediction == 1),
        'explanations': {
            'lime': explanations['lime'],
            'shap': explanations['shap'],
            'agreement': explanations['agreement'],
        },
        'fidelity': {
            'lime': lime_fidelity,
            'shap': shap_fidelity,
        },
    })


@app.route('/evaluate_adversarial', methods=['POST'])
def evaluate_adversarial():
    """
    Test a message against all adversarial attacks and compare model responses.
    Returns per-attack, per-model results with explanations.
    """
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    original = data['message']
    # Optionally specify which attacks to run
    requested_attacks = data.get('attacks', list(ATTACK_REGISTRY.keys()))

    results = {
        'original_text': original,
        'attacks': {},
    }

    models_to_test = []
    if bert_model and bert_classifier:
        models_to_test.append('bert')
    if tfidf_model and vectorizer:
        models_to_test.append('tfidf')

    for attack_name in requested_attacks:
        if attack_name not in ATTACK_REGISTRY:
            continue

        perturbed = apply_attack(original, attack_name)
        attack_info = ATTACK_REGISTRY[attack_name]

        attack_result = {
            'perturbed_text': perturbed,
            'attack_level': attack_info['level'],
            'attack_description': attack_info['description'],
            'models': {},
        }

        for m in models_to_test:
            predict_fn = get_predict_fn(m)

            orig_proba = predict_fn([original])[0]
            adv_proba = predict_fn([perturbed])[0]

            orig_pred = int(orig_proba.argmax())
            adv_pred = int(adv_proba.argmax())

            # Quick LIME explanation for both
            orig_exp = explainer.explain_lime(original, predict_fn, num_features=3, num_samples=50)
            adv_exp = explainer.explain_lime(perturbed, predict_fn, num_features=3, num_samples=50)

            attack_result['models'][m] = {
                'original': {
                    'prediction': "Spam" if orig_pred == 1 else "Not Spam",
                    'spam_probability': round(float(orig_proba[1]) * 100, 2),
                    'key_features': orig_exp['features'][:3],
                },
                'adversarial': {
                    'prediction': "Spam" if adv_pred == 1 else "Not Spam",
                    'spam_probability': round(float(adv_proba[1]) * 100, 2),
                    'key_features': adv_exp['features'][:3],
                },
                'fooled': orig_pred != adv_pred,
                'confidence_shift': round(float(orig_proba[1] - adv_proba[1]) * 100, 2),
            }

        results['attacks'][attack_name] = attack_result

    return jsonify(results)


@app.route('/attacks', methods=['GET'])
def list_attacks():
    """List all available adversarial attack strategies."""
    attacks = {}
    for name, info in ATTACK_REGISTRY.items():
        attacks[name] = {
            'level': info['level'],
            'description': info['description'],
        }
    return jsonify(attacks)


@app.route('/health', methods=['GET'])
def health():
    available_models = []
    if tfidf_model is not None and vectorizer is not None:
        available_models.append('tfidf')
    if bert_model is not None and bert_classifier is not None:
        available_models.append('bert')

    return jsonify({
        'status': 'healthy',
        'available_models': available_models,
        'available_attacks': list(ATTACK_REGISTRY.keys()),
        'explainability_methods': ['LIME', 'SHAP'],
        'default_model': 'bert' if 'bert' in available_models else 'tfidf' if 'tfidf' in available_models else None,
    })


@app.route('/models', methods=['GET'])
def list_models():
    models_info = {
        'tfidf': {
            'name': 'TF-IDF + Naive Bayes',
            'accuracy': '95.27%',
            'speed': 'Very Fast',
            'available': tfidf_model is not None and vectorizer is not None,
        },
        'bert': {
            'name': 'BERT + Logistic Regression',
            'accuracy': '~97-98%',
            'speed': 'Moderate',
            'available': bert_model is not None and bert_classifier is not None,
        },
    }
    return jsonify(models_info)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
