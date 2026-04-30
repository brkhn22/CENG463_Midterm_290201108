"""
Threshold Tuning and Cost Analysis
Analyzes precision-recall trade-offs and finds optimal decision threshold
"""

import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, auc, f1_score, confusion_matrix,
    ConfusionMatrixDisplay, precision_score, recall_score
)
import warnings

warnings.filterwarnings('ignore')

# Outputs are organized in outputs/q2/ subfolder
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'q2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Disable interactive plot display to keep pipeline running
SHOW_PLOTS = False


class ThresholdTuner:
    """
    Performs threshold tuning and cost analysis for binary classification
    """
    
    def __init__(self, y_true, y_pred_proba, model_name='Model'):
        """
        Initialize the threshold tuner
        
        Parameters:
        -----------
        y_true : array-like of shape (n_samples,)
            True binary labels
        y_pred_proba : array-like of shape (n_samples,)
            Predicted probabilities for the positive class
        model_name : str
            Name of the model for reporting
        """
        self.y_true = np.asarray(y_true)
        self.y_pred_proba = np.asarray(y_pred_proba)
        self.model_name = model_name
        
        self.precision = None
        self.recall = None
        self.thresholds = None
        self.pr_auc = None
        self.optimal_threshold = None
        self.optimal_f1 = None
        
    def calculate_pr_curve(self):
        """
        Calculate precision-recall curve
        """
        print(f"\nCalculating Precision-Recall curve for {self.model_name}...")
        
        self.precision, self.recall, self.thresholds = precision_recall_curve(
            self.y_true, self.y_pred_proba
        )
        
        # Calculate PR-AUC
        self.pr_auc = auc(self.recall, self.precision)
        
        print(f"PR-AUC: {self.pr_auc:.6f}")
        print(f"Number of thresholds: {len(self.thresholds)}")
        
        return self.precision, self.recall, self.thresholds
    
    def find_optimal_threshold(self):
        """
        Find the threshold that maximizes F1-score
        """
        if self.precision is None:
            self.calculate_pr_curve()
        
        print(f"\nFinding optimal threshold for {self.model_name}...")
        
        # Calculate F1-scores for all thresholds (excluding the last precision value)
        # Note: thresholds array has length = len(precision) - 1
        f1_scores = 2 * (self.precision[:-1] * self.recall[:-1]) / (
            self.precision[:-1] + self.recall[:-1] + 1e-10
        )
        
        # Find the threshold with maximum F1-score
        optimal_idx = np.argmax(f1_scores)
        self.optimal_threshold = self.thresholds[optimal_idx]
        self.optimal_f1 = f1_scores[optimal_idx]
        
        optimal_precision = self.precision[optimal_idx]
        optimal_recall = self.recall[optimal_idx]
        
        print(f"Optimal Threshold: {self.optimal_threshold:.6f}")
        print(f"Optimal F1-Score: {self.optimal_f1:.6f}")
        print(f"Precision at Optimal Threshold: {optimal_precision:.6f}")
        print(f"Recall at Optimal Threshold: {optimal_recall:.6f}")
        
        return self.optimal_threshold, self.optimal_f1
    
    def plot_pr_curve(self, save_path=None):
        """
        Plot and save the Precision-Recall curve
        
        Parameters:
        -----------
        save_path : str
            Path to save the plot
        """
        if save_path is None:
            save_path = os.path.join(OUTPUT_DIR, 'pr_curve.png')

        if self.precision is None:
            self.calculate_pr_curve()
        
        if self.optimal_threshold is None:
            self.find_optimal_threshold()
        
        print(f"\nPlotting Precision-Recall curve...")
        
        plt.figure(figsize=(12, 7))
        
        # Plot PR curve
        plt.plot(
            self.recall, self.precision, 
            linewidth=2.5, label=f'{self.model_name} (AUC = {self.pr_auc:.4f})',
            color='#1f77b4'
        )
        
        # Find the optimal point on the curve
        optimal_idx = np.argmax(
            2 * (self.precision[:-1] * self.recall[:-1]) / 
            (self.precision[:-1] + self.recall[:-1] + 1e-10)
        )
        optimal_precision = self.precision[optimal_idx]
        optimal_recall = self.recall[optimal_idx]
        
        # Mark the optimal threshold point
        plt.scatter(
            optimal_recall, optimal_precision,
            s=200, c='red', marker='*', 
            label=f'Optimal Threshold = {self.optimal_threshold:.4f}',
            zorder=5, edgecolors='darkred', linewidths=1.5
        )
        
        # Add baseline (no skill classifier)
        baseline_precision = np.sum(self.y_true) / len(self.y_true)
        plt.axhline(y=baseline_precision, color='gray', linestyle='--', linewidth=1.5,
                   label=f'Baseline (Prevalence = {baseline_precision:.4f})')
        
        # Formatting
        plt.xlabel('Recall', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        plt.title(f'{self.model_name}: Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved as '{save_path}'")
        
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()
    
    def apply_optimal_threshold(self):
        """
        Apply the optimal threshold to get final predictions
        
        Returns:
        --------
        y_pred_optimal : array-like
            Binary predictions using optimal threshold
        """
        if self.optimal_threshold is None:
            self.find_optimal_threshold()
        
        y_pred_optimal = (self.y_pred_proba >= self.optimal_threshold).astype(int)
        
        print(f"\nApplying optimal threshold ({self.optimal_threshold:.6f})...")
        print(f"Predicted positive class: {np.sum(y_pred_optimal)} samples")
        print(f"Predicted negative class: {len(y_pred_optimal) - np.sum(y_pred_optimal)} samples")
        
        return y_pred_optimal
    
    def plot_confusion_matrix(self, y_pred_optimal=None, save_path=None):
        """
        Plot and save confusion matrix using optimal threshold
        
        Parameters:
        -----------
        y_pred_optimal : array-like, optional
            Predictions using optimal threshold. If None, will be calculated.
        save_path : str
            Path to save the plot
        """
        if save_path is None:
            save_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_optimal.png')

        if y_pred_optimal is None:
            y_pred_optimal = self.apply_optimal_threshold()
        
        print(f"\nGenerating confusion matrix...")
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_true, y_pred_optimal)
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = sensitivity
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nConfusion Matrix Metrics:")
        print(f"  True Negatives (TN): {tn}")
        print(f"  False Positives (FP): {fp}")
        print(f"  False Negatives (FN): {fn}")
        print(f"  True Positives (TP): {tp}")
        print(f"\nPerformance Metrics:")
        print(f"  Precision: {precision:.6f}")
        print(f"  Recall (Sensitivity): {recall:.6f}")
        print(f"  Specificity: {specificity:.6f}")
        print(f"  F1-Score: {f1:.6f}")
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        
        # Add title with threshold information
        plt.title(
            f'{self.model_name}: Confusion Matrix (Optimal Threshold = {self.optimal_threshold:.4f})',
            fontsize=14, fontweight='bold', pad=20
        )
        
        # Add metrics text box
        metrics_text = f'Precision: {precision:.4f}\nRecall: {recall:.4f}\nSpecificity: {specificity:.4f}\nF1-Score: {f1:.4f}'
        ax.text(0.95, 0.05, metrics_text, transform=ax.transAxes,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=10, family='monospace')
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved as '{save_path}'")
        
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()
        
        return cm, {
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1
        }
    
    def plot_threshold_analysis(self, save_path=None):
        """
        Plot how precision, recall, and F1 vary with threshold
        
        Parameters:
        -----------
        save_path : str
            Path to save the plot
        """
        if save_path is None:
            save_path = os.path.join(OUTPUT_DIR, 'threshold_analysis.png')

        if self.precision is None:
            self.calculate_pr_curve()
        
        if self.optimal_threshold is None:
            self.find_optimal_threshold()
        
        print(f"\nPlotting threshold analysis...")
        
        # Calculate F1-scores for all thresholds
        f1_scores = 2 * (self.precision[:-1] * self.recall[:-1]) / (
            self.precision[:-1] + self.recall[:-1] + 1e-10
        )
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot precision, recall, and F1
        ax.plot(self.thresholds, self.precision[:-1], 'o-', linewidth=2, markersize=4,
               label='Precision', color='#1f77b4', alpha=0.7)
        ax.plot(self.thresholds, self.recall[:-1], 's-', linewidth=2, markersize=4,
               label='Recall', color='#ff7f0e', alpha=0.7)
        ax.plot(self.thresholds, f1_scores, '^-', linewidth=2, markersize=4,
               label='F1-Score', color='#2ca02c', alpha=0.7)
        
        # Mark optimal threshold
        ax.axvline(x=self.optimal_threshold, color='red', linestyle='--', linewidth=2.5,
                  label=f'Optimal Threshold = {self.optimal_threshold:.4f}')
        
        # Formatting
        ax.set_xlabel('Decision Threshold', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.model_name}: Precision, Recall, and F1 vs Threshold',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Threshold analysis plot saved as '{save_path}'")
        
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()
    
    def generate_summary_report(self):
        """
        Generate a comprehensive summary report
        """
        print("\n" + "=" * 70)
        print(f"THRESHOLD TUNING SUMMARY REPORT - {self.model_name}")
        print("=" * 70)
        
        if self.optimal_threshold is None:
            self.find_optimal_threshold()
        
        print(f"\nPrecision-Recall Curve:")
        print(f"  PR-AUC: {self.pr_auc:.6f}")
        print(f"\nOptimal Threshold Analysis:")
        print(f"  Optimal Threshold: {self.optimal_threshold:.6f}")
        print(f"  Maximum F1-Score: {self.optimal_f1:.6f}")
        print(f"  Default Threshold (0.5): {0.5:.6f}")
        
        # Compare with default threshold
        y_pred_default = (self.y_pred_proba >= 0.5).astype(int)
        f1_default = f1_score(self.y_true, y_pred_default)
        
        print(f"\nComparison with Default Threshold (0.5):")
        print(f"  F1-Score at default threshold: {f1_default:.6f}")
        print(f"  F1-Score at optimal threshold: {self.optimal_f1:.6f}")
        print(f"  Improvement: {(self.optimal_f1 - f1_default):.6f}")


def threshold_tuning_analysis(y_true, y_pred_proba, model_name='Model',
                             plot_pr=True, plot_confusion=True, plot_threshold=True):
    """
    Perform complete threshold tuning and cost analysis
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities for positive class
    model_name : str
        Name of the model
    plot_pr : bool
        Whether to plot PR curve
    plot_confusion : bool
        Whether to plot confusion matrix
    plot_threshold : bool
        Whether to plot threshold analysis
    
    Returns:
    --------
    tuner : ThresholdTuner
        The tuner instance with all analysis results
    y_pred_optimal : array-like
        Final predictions using optimal threshold
    """
    tuner = ThresholdTuner(y_true, y_pred_proba, model_name=model_name)
    
    # Calculate PR curve
    tuner.calculate_pr_curve()
    
    # Find optimal threshold
    tuner.find_optimal_threshold()
    
    # Plot PR curve
    if plot_pr:
        tuner.plot_pr_curve()
    
    # Apply optimal threshold
    y_pred_optimal = tuner.apply_optimal_threshold()
    
    # Plot confusion matrix
    if plot_confusion:
        cm, metrics = tuner.plot_confusion_matrix(y_pred_optimal=y_pred_optimal)
    
    # Plot threshold analysis
    if plot_threshold:
        tuner.plot_threshold_analysis()
    
    # Generate summary report
    tuner.generate_summary_report()
    
    return tuner, y_pred_optimal


if __name__ == "__main__":
    # Example usage with synthetic data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    print("Running threshold tuning as standalone script...")
    print("Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        weights=[0.9, 0.1],
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print("\nTraining RandomForest classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Get predicted probabilities
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Run threshold tuning analysis
    print("\n" + "=" * 70)
    tuner, y_pred_optimal = threshold_tuning_analysis(
        y_test, y_pred_proba,
        model_name='RandomForest',
        plot_pr=True,
        plot_confusion=True,
        plot_threshold=True
    )
