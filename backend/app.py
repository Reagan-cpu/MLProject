from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load TF-IDF Model
TFIDF_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'spam_model.joblib')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'vectorizer.joblib')

tfidf_model = None
vectorizer = None

if os.path.exists(TFIDF_MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    tfidf_model = joblib.load(TFIDF_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

# Load BERT Model
BERT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'bert_model.joblib')
BERT_CLASSIFIER_PATH = os.path.join(os.path.dirname(__file__), 'spam_model_bert.joblib')

bert_model = None
bert_classifier = None

if os.path.exists(BERT_MODEL_PATH) and os.path.exists(BERT_CLASSIFIER_PATH):
    bert_model = joblib.load(BERT_MODEL_PATH)
    bert_classifier = joblib.load(BERT_CLASSIFIER_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    
    # Default to BERT if available, otherwise use TF-IDF
    model_type = data.get('model', 'bert' if bert_model else 'tfidf')
    message = data['message']
    
    # BERT Model
    if model_type == 'bert':
        if bert_model is None or bert_classifier is None:
            return jsonify({'error': 'BERT model not trained. Run train_model_bert.py first.'}), 500
        
        embedding = bert_model.encode([message])
        prediction = bert_classifier.predict(embedding)[0]
        probabilities = bert_classifier.predict_proba(embedding)[0]
    
    # TF-IDF Model
    elif model_type == 'tfidf':
        if tfidf_model is None or vectorizer is None:
            return jsonify({'error': 'TF-IDF model not trained. Run train_model.py first.'}), 500
        
        message_tfidf = vectorizer.transform([message])
        prediction = tfidf_model.predict(message_tfidf)[0]
        probabilities = tfidf_model.predict_proba(message_tfidf)[0]
    
    else:
        return jsonify({'error': 'Invalid model type. Use "bert" or "tfidf".'}), 400
    
    # Result mapping
    label = "Spam" if prediction == 1 else "Not Spam"
    probability = float(probabilities[prediction])
    
    return jsonify({
        'prediction': label,
        'probability': round(probability * 100, 2),
        'is_spam': bool(prediction == 1),
        'model_used': model_type
    })

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
        'default_model': 'bert' if 'bert' in available_models else 'tfidf' if 'tfidf' in available_models else None
    })

@app.route('/models', methods=['GET'])
def list_models():
    models_info = {
        'tfidf': {
            'name': 'TF-IDF + Naive Bayes',
            'accuracy': '95.27%',
            'speed': 'Very Fast',
            'available': tfidf_model is not None and vectorizer is not None
        },
        'bert': {
            'name': 'BERT + Logistic Regression',
            'accuracy': '~97-98%',
            'speed': 'Moderate',
            'available': bert_model is not None and bert_classifier is not None
        }
    }
    return jsonify(models_info)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
