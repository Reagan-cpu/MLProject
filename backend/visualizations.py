"""
Visualization Module for Spam Detection Models
===============================================
Generates plots for model evaluation including:
  - Confusion matrices
  - ROC curves
  - Model performance comparison
  - Adversarial robustness analysis
  - Training metrics
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import json

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_confusion_matrix(y_true, y_pred, model_name, output_dir='reports'):
    """Plot and save confusion matrix heatmap."""
    os.makedirs(output_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f'confusion_matrix_{model_name.replace(" ", "_").replace("+", "").lower()}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filename}")


def plot_roc_curves(models_results, output_dir='reports'):
    """Plot ROC curves for multiple models."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # Define colors for different models
    colors = {'tfidf': 'blue', 'bert': 'red'}
    
    for model_key, metrics in models_results.items():
        # Plot diagonal (random classifier)
        if model_key == 'tfidf':
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier', alpha=0.5)
        
        auc_score = metrics.get('auc_roc', 0)
        model_name = metrics.get('name', model_key)
        plt.plot([], [], color=colors.get(model_key, 'gray'), lw=2.5, 
                label=f'{model_name} (AUC = {auc_score:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'roc_curves_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filename}")


def plot_metrics_comparison(models_results, output_dir='reports'):
    """Compare key metrics across models."""
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    model_names = []
    metrics_data = {metric: [] for metric in metrics_to_plot}
    
    for model_key, metrics in models_results.items():
        model_names.append(metrics.get('name', model_key))
        for metric in metrics_to_plot:
            metrics_data[metric].append(metrics.get(metric, 0))
    
    # Create grouped bar plot
    x = np.arange(len(model_names))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, metric in enumerate(metrics_to_plot):
        offset = (i - 2) * width
        ax.bar(x + offset, metrics_data[metric], width, label=metric.upper(), 
               color=colors_list[i], alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filename}")


def plot_adversarial_robustness(adversarial_results, output_dir='reports'):
    """Plot model robustness under adversarial attacks."""
    os.makedirs(output_dir, exist_ok=True)
    
    attacks = []
    evasion_rates_tfidf = []
    evasion_rates_bert = []
    
    for attack_name, attack_data in adversarial_results.items():
        attacks.append(attack_name.replace('_', ' ').title())
        
        tfidf_evasion = attack_data.get('models', {}).get('tfidf', {}).get('evasion_rate', 0)
        bert_evasion = attack_data.get('models', {}).get('bert', {}).get('evasion_rate', 0)
        
        evasion_rates_tfidf.append(tfidf_evasion)
        evasion_rates_bert.append(bert_evasion)
    
    x = np.arange(len(attacks))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width/2, evasion_rates_tfidf, width, label='TF-IDF + Naive Bayes', 
           color='#1f77b4', alpha=0.8)
    ax.bar(x + width/2, evasion_rates_bert, width, label='BERT + Logistic Regression', 
           color='#ff7f0e', alpha=0.8)
    
    ax.set_ylabel('Evasion Rate', fontsize=12)
    ax.set_title('Model Robustness Against Adversarial Attacks', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'adversarial_robustness.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filename}")


def plot_explainability_fidelity(explainability_results, output_dir='reports'):
    """Compare explanation fidelity across models and methods."""
    os.makedirs(output_dir, exist_ok=True)
    
    model_names = []
    lime_fidelity = []
    shap_fidelity = []
    agreement_scores = []
    
    for model_key, results in explainability_results.items():
        model_names.append(results.get('model_name', model_key).replace('_', ' ').title())
        lime_fidelity.append(results.get('avg_fidelity_lime', 0) or 0)
        shap_fidelity.append(results.get('avg_fidelity_shap', 0) or 0)
        agreement_scores.append(results.get('avg_agreement', 0) or 0)
    
    x = np.arange(len(model_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width, lime_fidelity, width, label='LIME Fidelity', color='#1f77b4', alpha=0.8)
    ax.bar(x, shap_fidelity, width, label='SHAP Fidelity', color='#ff7f0e', alpha=0.8)
    ax.bar(x + width, agreement_scores, width, label='LIME-SHAP Agreement', color='#2ca02c', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Explanation Fidelity & Agreement Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'explainability_fidelity.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filename}")


def generate_summary_report(all_results, output_dir='reports'):
    """Generate a comprehensive summary figure with all key metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Accuracy comparison
    ax1 = fig.add_subplot(gs[0, 0])
    models = list(all_results['clean_metrics'].keys())
    accuracies = [all_results['clean_metrics'][m]['accuracy'] for m in models]
    colors = ['#1f77b4', '#ff7f0e']
    ax1.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Comparison')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. F1-Score comparison
    ax2 = fig.add_subplot(gs[0, 1])
    f1_scores = [all_results['clean_metrics'][m]['f1'] for m in models]
    ax2.bar(models, f1_scores, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('F1-Score Comparison')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Precision vs Recall
    ax3 = fig.add_subplot(gs[1, 0])
    precisions = [all_results['clean_metrics'][m]['precision'] for m in models]
    recalls = [all_results['clean_metrics'][m]['recall'] for m in models]
    x = np.arange(len(models))
    width = 0.35
    ax3.bar(x - width/2, precisions, width, label='Precision', color='#2ca02c', alpha=0.8)
    ax3.bar(x + width/2, recalls, width, label='Recall', color='#d62728', alpha=0.8)
    ax3.set_ylabel('Score')
    ax3.set_title('Precision vs Recall')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.set_ylim([0, 1])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. AUC-ROC comparison
    ax4 = fig.add_subplot(gs[1, 1])
    aucs = [all_results['clean_metrics'][m]['auc_roc'] for m in models]
    ax4.bar(models, aucs, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_ylabel('AUC-ROC')
    ax4.set_title('AUC-ROC Comparison')
    ax4.set_ylim([0, 1])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Robustness summary (average evasion rate)
    ax5 = fig.add_subplot(gs[2, 0])
    if 'adversarial_metrics' in all_results:
        avg_evasion_tfidf = []
        avg_evasion_bert = []
        for attack_name, attack_data in all_results['adversarial_metrics'].items():
            tfidf_evasion = attack_data.get('models', {}).get('tfidf', {}).get('evasion_rate', 0)
            bert_evasion = attack_data.get('models', {}).get('bert', {}).get('evasion_rate', 0)
            avg_evasion_tfidf.append(tfidf_evasion)
            avg_evasion_bert.append(bert_evasion)
        
        model_robustness = ['TF-IDF + NB', 'BERT + LR']
        avg_evasion = [np.mean(avg_evasion_tfidf), np.mean(avg_evasion_bert)]
        ax5.bar(model_robustness, avg_evasion, color=colors, alpha=0.8, edgecolor='black')
        ax5.set_ylabel('Avg Evasion Rate')
        ax5.set_title('Average Robustness Against Attacks')
        ax5.set_ylim([0, 1])
        ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Summary text
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    summary_text = "Model Evaluation Summary\n" + "="*30 + "\n\n"
    for model in models:
        metrics = all_results['clean_metrics'][model]
        summary_text += f"{model}:\n"
        summary_text += f"  Accuracy: {metrics['accuracy']:.3f}\n"
        summary_text += f"  F1-Score: {metrics['f1']:.3f}\n"
        summary_text += f"  AUC-ROC: {metrics['auc_roc']:.3f}\n\n"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
            fontfamily='monospace', fontsize=9, verticalalignment='top')
    
    plt.suptitle('Comprehensive Model Evaluation Report', fontsize=16, fontweight='bold', y=0.995)
    
    filename = os.path.join(output_dir, 'summary_report.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filename}")


if __name__ == "__main__":
    print("Visualizations module loaded successfully")
