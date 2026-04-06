"""
Explainability Module for Spam Detection Research
===================================================
Provides unified interface for multiple explanation methods:
  - LIME (Local Interpretable Model-agnostic Explanations)
  - SHAP (SHapley Additive exPlanations)
  - Feature Importance (model-specific)

Each explainer returns a standardized format for cross-method comparison.
"""

import numpy as np
from lime.lime_text import LimeTextExplainer

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class UnifiedExplainer:
    """
    Unified explainability interface for comparing LIME vs SHAP.
    
    Returns standardized explanation format:
    {
        "method": str,
        "features": [{"word": str, "weight": float, "direction": "spam"|"ham"}, ...],
        "top_spam_words": [str, ...],
        "top_ham_words": [str, ...],
    }
    """

    def __init__(self, class_names=None):
        self.class_names = class_names or ["Not Spam", "Spam"]
        self.lime_explainer = LimeTextExplainer(
            class_names=self.class_names,
            split_expression=r'\W+',
            bow=True,
        )

    def _standardize(self, method, feature_weights):
        """Convert raw feature weights to standardized format."""
        features = []
        top_spam = []
        top_ham = []

        for word, weight in sorted(feature_weights, key=lambda x: abs(x[1]), reverse=True):
            direction = "spam" if weight > 0 else "ham"
            features.append({
                "word": word,
                "weight": round(float(weight), 6),
                "direction": direction,
            })
            if direction == "spam":
                top_spam.append(word)
            else:
                top_ham.append(word)

        return {
            "method": method,
            "features": features,
            "top_spam_words": top_spam[:5],
            "top_ham_words": top_ham[:5],
        }

    def explain_lime(self, text, predict_fn, num_features=8, num_samples=200):
        """Generate LIME explanation."""
        exp = self.lime_explainer.explain_instance(
            text,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=[1],  # Explain the "Spam" class
        )
        feature_weights = exp.as_list(label=1)
        result = self._standardize("LIME", feature_weights)
        
        # Add LIME-specific: prediction probabilities from the explanation
        probs = exp.predict_proba
        if probs is not None:
            result["predicted_proba"] = {
                "ham": round(float(probs[0]), 4),
                "spam": round(float(probs[1]), 4),
            }
        
        return result

    def explain_shap(self, text, predict_fn, num_samples=100):
        """Generate SHAP explanation using KernelExplainer."""
        if not SHAP_AVAILABLE:
            return {
                "method": "SHAP",
                "error": "shap package not installed. Run: pip install shap",
                "features": [],
                "top_spam_words": [],
                "top_ham_words": [],
            }

        words = text.split()
        if len(words) == 0:
            return self._standardize("SHAP", [])

        # Build a simple BoW-style masker for SHAP
        def shap_predict(masks):
            """Predict function that accepts binary masks over words."""
            texts = []
            for mask in masks:
                masked_words = [w for w, m in zip(words, mask) if m]
                texts.append(" ".join(masked_words) if masked_words else " ")
            probs = predict_fn(texts)
            return probs

        # Create background of all-ones (full text present)
        background = np.ones((1, len(words)))

        try:
            explainer_kernel = shap.KernelExplainer(shap_predict, background)
            shap_values = explainer_kernel.shap_values(
                np.ones((1, len(words))),
                nsamples=num_samples,
            )

            # shap_values shape: (n_classes, n_samples, n_features) or (n_samples, n_features)
            if isinstance(shap_values, list):
                # Multi-class: take the "Spam" class (index 1)
                sv = shap_values[1][0]
            else:
                sv = shap_values[0]

            feature_weights = list(zip(words, sv.tolist()))
            return self._standardize("SHAP", feature_weights)

        except Exception as e:
            return {
                "method": "SHAP",
                "error": str(e),
                "features": [],
                "top_spam_words": [],
                "top_ham_words": [],
            }

    def explain_all(self, text, predict_fn, num_features=8):
        """Run all available explanation methods and return comparative results."""
        results = {}

        # LIME
        results["lime"] = self.explain_lime(text, predict_fn, num_features=num_features)

        # SHAP
        results["shap"] = self.explain_shap(text, predict_fn)

        # Agreement analysis: do LIME and SHAP agree on top features?
        lime_top = set(results["lime"]["top_spam_words"][:3])
        shap_top = set(results["shap"].get("top_spam_words", [])[:3])

        if lime_top and shap_top:
            overlap = lime_top & shap_top
            agreement = len(overlap) / max(len(lime_top), len(shap_top))
        else:
            agreement = None

        results["agreement"] = {
            "lime_top3_spam": list(lime_top),
            "shap_top3_spam": list(shap_top),
            "overlap": list(lime_top & shap_top) if lime_top and shap_top else [],
            "agreement_score": round(agreement, 2) if agreement is not None else None,
        }

        return results

    def compute_explanation_fidelity(self, text, predict_fn, explanation, top_k=3):
        """
        Measure explanation fidelity by removing top-k important words
        and checking if the prediction changes.
        
        A faithful explanation should cause the prediction to flip
        when its most important features are removed.
        """
        words = text.split()
        original_proba = predict_fn([text])[0]
        original_pred = int(np.argmax(original_proba))
        original_spam_prob = float(original_proba[1])

        # Get top-k words to remove (highest absolute weight)
        important_words = [f["word"] for f in explanation["features"][:top_k]]

        # Remove important words
        filtered = [w for w in words if w.lower() not in [iw.lower() for iw in important_words]]
        reduced_text = " ".join(filtered) if filtered else " "

        reduced_proba = predict_fn([reduced_text])[0]
        reduced_pred = int(np.argmax(reduced_proba))
        reduced_spam_prob = float(reduced_proba[1])

        return {
            "original_spam_prob": round(original_spam_prob, 4),
            "reduced_spam_prob": round(reduced_spam_prob, 4),
            "probability_drop": round(abs(original_spam_prob - reduced_spam_prob), 4),
            "prediction_flipped": original_pred != reduced_pred,
            "removed_words": important_words,
            "fidelity_score": round(abs(original_spam_prob - reduced_spam_prob) / max(original_spam_prob, 0.01), 4),
        }
