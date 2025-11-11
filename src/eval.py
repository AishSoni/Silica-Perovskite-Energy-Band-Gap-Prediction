"""
Evaluation module with metrics, visualizations, and SHAP analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import warnings

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

warnings.filterwarnings('ignore')

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization.
    """
    
    def __init__(self, output_dir: str = "figures"):
        """
        Initialize evaluator.
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def regression_metrics(self, y_true, y_pred) -> Dict:
        """
        Compute regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            Dictionary with metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        errors = np.abs(y_true - y_pred)
        median_ae = np.median(errors)
        max_error = np.max(errors)
        
        # Percentage of samples with >25% error
        pct_error = errors / (y_true + 1e-10) * 100
        pct_over_25 = (pct_error > 25).sum() / len(pct_error) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MSE': mse,
            'R²': r2,
            'Median_AE': median_ae,
            'Max_Error': max_error,
            'Pct_Error_>25%': pct_over_25
        }
        
        return metrics
    
    def classification_metrics(self, y_true, y_pred, y_pred_proba=None) -> Dict:
        """
        Compute classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
        
        Returns:
            Dictionary with metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
        
        # Add ROC-AUC if probabilities available
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:  # Multiclass
                    roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                metrics['ROC-AUC'] = roc_auc
            except:
                pass
        
        return metrics
    
    def parity_plot(
        self,
        y_true,
        y_pred,
        title: str = "Parity Plot",
        filename: str = "parity_plot.png",
        show_metrics: bool = True
    ):
        """
        Create parity plot for regression.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            filename: Output filename
            show_metrics: Whether to show metrics on plot
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, s=20, alpha=0.5, edgecolors='k', linewidths=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        # Labels and title
        ax.set_xlabel('DFT Bandgap (eV)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Bandgap (eV)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add metrics
        if show_metrics:
            metrics = self.regression_metrics(y_true, y_pred)
            text = f"MAE: {metrics['MAE']:.3f} eV\nRMSE: {metrics['RMSE']:.3f} eV\nR²: {metrics['R²']:.3f}"
            ax.text(0.05, 0.95, text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=10)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Parity plot saved to {output_path}")
    
    def error_histogram(
        self,
        y_true,
        y_pred,
        title: str = "Prediction Error Distribution",
        filename: str = "error_histogram.png"
    ):
        """
        Create error distribution histogram.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            filename: Output filename
        """
        errors = y_pred - y_true
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
        
        ax.set_xlabel('Prediction Error (eV)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        text = f"Mean: {errors.mean():.3f}\nStd: {errors.std():.3f}\nMedian: {np.median(errors):.3f}"
        ax.text(0.95, 0.95, text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Error histogram saved to {output_path}")
    
    def confusion_matrix_plot(
        self,
        y_true,
        y_pred,
        labels: Optional[List] = None,
        title: str = "Confusion Matrix",
        filename: str = "confusion_matrix.png"
    ):
        """
        Create confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            title: Plot title
            filename: Output filename
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Use labels if provided, otherwise use auto labels from confusion matrix
        if labels is None:
            labels = ['False', 'True']  # default binary labels
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=labels, yticklabels=labels)
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrix saved to {output_path}")
    
    def roc_curve_plot(
        self,
        y_true,
        y_pred_proba,
        title: str = "ROC Curve",
        filename: str = "roc_curve.png"
    ):
        """
        Create ROC curve plot.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            filename: Output filename
        """
        # Binary classification
        if len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
            
            ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            output_path = self.output_dir / filename
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ ROC curve saved to {output_path}")
    
    def feature_importance_plot(
        self,
        model,
        feature_names: List[str],
        top_n: int = 20,
        title: str = "Feature Importance",
        filename: str = "feature_importance.png"
    ):
        """
        Create feature importance bar plot.
        
        Args:
            model: Trained model with feature_importances_
            feature_names: List of feature names
            top_n: Number of top features to show
            title: Plot title
            filename: Output filename
        """
        if not hasattr(model, 'feature_importances_'):
            print("⚠ Model does not have feature_importances_ attribute")
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:][::-1]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(top_n)
        ax.barh(y_pos, importances[indices], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Feature importance plot saved to {output_path}")
    
    def shap_analysis(
        self,
        model,
        X,
        feature_names: List[str],
        max_display: int = 20,
        prefix: str = "shap"
    ):
        """
        Perform SHAP analysis and create visualizations.
        
        Args:
            model: Trained model
            X: Feature data
            feature_names: List of feature names
            max_display: Maximum features to display
            prefix: Prefix for output filenames
        """
        if not SHAP_AVAILABLE:
            print("⚠ SHAP not available, skipping SHAP analysis")
            return
        
        print("Computing SHAP values (this may take a while)...")
        
        try:
            # Fix XGBoost base_score serialization issue for SHAP
            if hasattr(model, 'get_booster'):
                # It's an XGBoost model - aggressively fix base_score in JSON config
                try:
                    import json
                    import tempfile
                    import os
                    
                    # Get the booster's JSON config
                    booster = model.get_booster()
                    config_json = booster.save_config()
                    config_dict = json.loads(config_json)
                    
                    # Fix base_score in the config (change from '[5E-1]' string to '0.5' string)
                    if 'learner' in config_dict and 'learner_model_param' in config_dict['learner']:
                        params = config_dict['learner']['learner_model_param']
                        if 'base_score' in params:
                            # Convert any string representation to numeric string '0.5'
                            old_val = params['base_score']
                            params['base_score'] = '0.5'
                            print(f"  Fixed base_score: {old_val} → 0.5")
                    
                    # Reload the fixed config back into the booster
                    booster.load_config(json.dumps(config_dict))
                    
                    # Also ensure the model attribute is numeric
                    if hasattr(model, 'base_score'):
                        model.base_score = 0.5
                        
                except Exception as fix_error:
                    print(f"  Note: Could not fix XGBoost config ({fix_error}), proceeding anyway...")
            
            # Try TreeExplainer first (fast for tree models)
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            except Exception as tree_error:
                # If TreeExplainer fails, fall back to generic Explainer (slower but more robust)
                if "could not convert string to float" in str(tree_error):
                    print(f"  TreeExplainer failed due to base_score issue, trying generic Explainer...")
                    explainer = shap.Explainer(model.predict, X)
                    shap_values = explainer(X).values
                else:
                    raise tree_error
            
            # Summary plot
            plt.figure()
            shap.summary_plot(shap_values, X, feature_names=feature_names, 
                            max_display=max_display, show=False)
            output_path = self.output_dir / f"{prefix}_summary.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ SHAP summary plot saved to {output_path}")
            
            # Bar plot
            plt.figure()
            shap.summary_plot(shap_values, X, feature_names=feature_names,
                            max_display=max_display, plot_type='bar', show=False)
            output_path = self.output_dir / f"{prefix}_bar.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ SHAP bar plot saved to {output_path}")
            
        except Exception as e:
            print(f"⚠ SHAP analysis failed: {e}")
    
    def compare_models(
        self,
        results_dict: Dict[str, Dict],
        task: str = 'regression',
        filename: str = "model_comparison.png"
    ):
        """
        Create comparison plot for multiple models.
        
        Args:
            results_dict: Dictionary with model names and their metrics
            task: 'regression' or 'classification'
            filename: Output filename
        """
        if task == 'regression':
            metrics_to_plot = ['MAE', 'RMSE', 'R²']
        else:
            metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        # Prepare data
        model_names = list(results_dict.keys())
        n_metrics = len(metrics_to_plot)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics_to_plot):
            values = [results_dict[model].get(metric, 0) for model in model_names]
            
            axes[idx].bar(range(len(model_names)), values, color='steelblue', edgecolor='black')
            axes[idx].set_xticks(range(len(model_names)))
            axes[idx].set_xticklabels(model_names, rotation=45, ha='right')
            axes[idx].set_ylabel(metric, fontsize=12, fontweight='bold')
            axes[idx].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Model comparison plot saved to {output_path}")
    
    def save_results(self, results: Dict, filename: str = "evaluation_results.csv"):
        """
        Save evaluation results to CSV.
        
        Args:
            results: Dictionary with results
            filename: Output filename
        """
        df = pd.DataFrame(results).T
        output_path = self.output_dir.parent / "results" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path)
        print(f"✓ Results saved to {output_path}")


def evaluate_model(
    model_path: str,
    X_test_path: str,
    y_test_path: str,
    task: str = 'regression',
    output_dir: str = 'figures',
    feature_names_path: Optional[str] = None
) -> Dict:
    """
    Main function to evaluate a trained model.
    
    Args:
        model_path: Path to saved model
        X_test_path: Path to test features
        y_test_path: Path to test target
        task: 'regression' or 'classification'
        output_dir: Directory to save figures
        feature_names_path: Path to feature names file
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*80)
    print(f"MODEL EVALUATION - {task.upper()}")
    print("="*80 + "\n")
    
    # Load model and data
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    print(f"Loading test data...")
    X_test = joblib.load(X_test_path)
    y_test = joblib.load(y_test_path)
    
    print(f"✓ Test set: {X_test.shape}")
    
    # Load feature names
    feature_names = None
    if feature_names_path:
        with open(feature_names_path, 'r') as f:
            feature_names = [line.strip() for line in f]
    
    # Initialize evaluator
    evaluator = ModelEvaluator(output_dir=output_dir)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    
    # Compute metrics
    if task == 'regression':
        metrics = evaluator.regression_metrics(y_test, y_pred)
        
        print("\nRegression Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Create visualizations
        evaluator.parity_plot(y_test, y_pred)
        evaluator.error_histogram(y_test, y_pred)
        
    else:  # classification
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        metrics = evaluator.classification_metrics(y_test, y_pred, y_pred_proba)
        
        print("\nClassification Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Create visualizations
        evaluator.confusion_matrix_plot(y_test, y_pred)
        if y_pred_proba is not None:
            evaluator.roc_curve_plot(y_test, y_pred_proba)
    
    # Feature importance
    if feature_names and hasattr(model, 'feature_importances_'):
        evaluator.feature_importance_plot(model, feature_names)
    
    # SHAP analysis
    if SHAP_AVAILABLE and feature_names:
        evaluator.shap_analysis(model, X_test[:100], feature_names)  # Use subset for speed
    
    print("\n✓ Evaluation complete!")
    
    return metrics


if __name__ == "__main__":
    print("Testing evaluation module...")
    
    # This requires trained model and test data
    print("Note: This test requires trained model files to exist")
