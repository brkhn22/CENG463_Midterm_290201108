"""
Probability Calibration Analysis
Compares calibrated vs uncalibrated classifiers using Brier score and reliability diagrams
"""

import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import brier_score_loss
import warnings

warnings.filterwarnings('ignore')

# Outputs are organized in outputs/q2/ subfolder
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'q2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Disable interactive plot display to keep pipeline running
SHOW_PLOTS = False


class CalibrationAnalyzer:
    """
    Analyzes and compares probability calibration for trained classifiers
    """
    
    def __init__(self, models_dict, X_train, X_test, y_train, y_test):
        """
        Initialize the calibration analyzer
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary with model names as keys and trained model instances as values
            Expected keys: 'XGBoost', 'RandomForest'
        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix
        X_test : array-like of shape (n_samples, n_features)
            Test feature matrix
        y_train : array-like of shape (n_samples,)
            Training target variable
        y_test : array-like of shape (n_samples,)
            Test target variable
        """
        self.models_dict = models_dict
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.calibrated_models = {}
        self.brier_scores = {}
        self.predictions = {}
        
    def _calibrate_model(self, model, model_name, method='sigmoid'):
        """
        Apply probability calibration to a model
        
        Parameters:
        -----------
        model : estimator
            Trained classifier
        model_name : str
            Name of the model
        method : str, default='sigmoid'
            Calibration method: 'sigmoid' (Platt scaling) or 'isotonic'
        
        Returns:
        --------
        calibrated_model : CalibratedClassifierCV
            Calibrated model instance
        """
        print(f"Calibrating {model_name} with {method} method...")
        
        # Calibrate with CV to avoid deprecated 'prefit' mode in newer sklearn
        calibrated_model = CalibratedClassifierCV(
            estimator=clone(model),
            method=method,
            cv=3
        )
        
        # Fit the calibrator on training data
        calibrated_model.fit(self.X_train, self.y_train)
        
        return calibrated_model
    
    def _evaluate_brier_score(self, model, y_true, set_name='Test'):
        """
        Calculate Brier score loss for a model
        
        Parameters:
        -----------
        model : estimator
            Trained classifier
        y_true : array-like
            True labels
        set_name : str
            Name of the dataset ('Train' or 'Test')
        
        Returns:
        --------
        brier_score : float
            Brier score loss
        """
        y_pred_proba = model.predict_proba(self.X_test if set_name == 'Test' else self.X_train)[:, 1]
        brier_score = brier_score_loss(y_true, y_pred_proba)
        return brier_score
    
    def analyze_calibration(self):
        """
        Perform calibration analysis for all models
        """
        print("=" * 70)
        print("PROBABILITY CALIBRATION ANALYSIS")
        print("=" * 70)
        print()
        
        results = []
        
        for model_name, model in self.models_dict.items():
            print(f"\n{model_name}")
            print("-" * 70)
            
            # 1. Get uncalibrated predictions and Brier score
            print(f"Uncalibrated {model_name}:")
            y_pred_proba_uncalibrated = model.predict_proba(self.X_test)[:, 1]
            brier_uncalibrated = brier_score_loss(self.y_test, y_pred_proba_uncalibrated)
            print(f"  Brier Score (Uncalibrated): {brier_uncalibrated:.6f}")
            
            results.append({
                'Model': model_name,
                'Calibration': 'Uncalibrated',
                'Brier Score': brier_uncalibrated
            })
            
            self.predictions[f"{model_name}_Uncalibrated"] = y_pred_proba_uncalibrated
            
            # 2. Calibrate with sigmoid (Platt scaling)
            print(f"\nCalibrated {model_name} (Sigmoid/Platt Scaling):")
            calibrated_sigmoid = self._calibrate_model(model, model_name, method='sigmoid')
            y_pred_proba_sigmoid = calibrated_sigmoid.predict_proba(self.X_test)[:, 1]
            brier_sigmoid = brier_score_loss(self.y_test, y_pred_proba_sigmoid)
            print(f"  Brier Score (Sigmoid): {brier_sigmoid:.6f}")
            print(f"  Improvement: {brier_uncalibrated - brier_sigmoid:.6f}")
            
            results.append({
                'Model': model_name,
                'Calibration': 'Sigmoid (Platt)',
                'Brier Score': brier_sigmoid
            })
            
            self.predictions[f"{model_name}_Sigmoid"] = y_pred_proba_sigmoid
            self.calibrated_models[f"{model_name}_Sigmoid"] = calibrated_sigmoid
            
            # 3. Calibrate with isotonic
            print(f"\nCalibrated {model_name} (Isotonic):")
            try:
                calibrated_isotonic = self._calibrate_model(model, model_name, method='isotonic')
                y_pred_proba_isotonic = calibrated_isotonic.predict_proba(self.X_test)[:, 1]
                brier_isotonic = brier_score_loss(self.y_test, y_pred_proba_isotonic)
                print(f"  Brier Score (Isotonic): {brier_isotonic:.6f}")
                print(f"  Improvement: {brier_uncalibrated - brier_isotonic:.6f}")
                
                results.append({
                    'Model': model_name,
                    'Calibration': 'Isotonic',
                    'Brier Score': brier_isotonic
                })
                
                self.predictions[f"{model_name}_Isotonic"] = y_pred_proba_isotonic
                self.calibrated_models[f"{model_name}_Isotonic"] = calibrated_isotonic
                
            except Exception as e:
                print(f"  Warning: Isotonic calibration failed - {str(e)}")
                print(f"  This may occur when there are insufficient samples or class imbalance issues")
        
        return pd.DataFrame(results)
    
    def plot_calibration_curves(self):
        """
        Generate and save reliability diagrams (calibration curves)
        """
        print("\n" + "=" * 70)
        print("GENERATING CALIBRATION CURVES")
        print("=" * 70)
        
        # Create figure with subplots for each model
        n_models = len(self.models_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(15, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, ax) in enumerate(zip(self.models_dict.keys(), axes)):
            print(f"\nPlotting calibration curve for {model_name}...")
            
            # Plot uncalibrated model
            y_pred_uncalibrated = self.predictions[f"{model_name}_Uncalibrated"]
            disp = CalibrationDisplay.from_predictions(
                self.y_test,
                y_pred_uncalibrated,
                n_bins=10,
                strategy='uniform',
                pos_label=1,
                ax=ax,
                name='Uncalibrated'
            )
            
            # Plot sigmoid calibrated model
            y_pred_sigmoid = self.predictions[f"{model_name}_Sigmoid"]
            CalibrationDisplay.from_predictions(
                self.y_test,
                y_pred_sigmoid,
                n_bins=10,
                strategy='uniform',
                pos_label=1,
                ax=ax,
                name='Sigmoid (Platt)',
                marker='s'
            )
            
            # Plot isotonic calibrated model if available
            if f"{model_name}_Isotonic" in self.predictions:
                y_pred_isotonic = self.predictions[f"{model_name}_Isotonic"]
                CalibrationDisplay.from_predictions(
                    self.y_test,
                    y_pred_isotonic,
                    n_bins=10,
                    strategy='uniform',
                    pos_label=1,
                    ax=ax,
                    name='Isotonic',
                    marker='^'
                )
            
            ax.set_title(f'{model_name} Calibration Curve', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(OUTPUT_DIR, 'calibration_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nCalibration curves saved as '{output_path}'")
        
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()
    
    def plot_combined_calibration_curve(self):
        """
        Generate a single combined calibration curve for all models and methods
        """
        print("\n" + "=" * 70)
        print("GENERATING COMBINED CALIBRATION CURVE")
        print("=" * 70)
        
        plt.figure(figsize=(12, 8))
        
        # Define colors for different calibration methods
        colors = {
            'Uncalibrated': '#1f77b4',
            'Sigmoid': '#ff7f0e',
            'Isotonic': '#2ca02c'
        }
        
        markers = {
            'Uncalibrated': 'o',
            'Sigmoid': 's',
            'Isotonic': '^'
        }
        
        # Plot each model's predictions
        for model_name in self.models_dict.keys():
            print(f"\nPlotting {model_name}...")
            
            # Uncalibrated
            y_pred_uncalibrated = self.predictions[f"{model_name}_Uncalibrated"]
            disp = CalibrationDisplay.from_predictions(
                self.y_test,
                y_pred_uncalibrated,
                n_bins=10,
                strategy='uniform',
                pos_label=1,
                name=f'{model_name} (Uncalibrated)',
                marker=markers['Uncalibrated'],
                color=colors['Uncalibrated']
            )
            
            # Sigmoid calibrated
            y_pred_sigmoid = self.predictions[f"{model_name}_Sigmoid"]
            CalibrationDisplay.from_predictions(
                self.y_test,
                y_pred_sigmoid,
                n_bins=10,
                strategy='uniform',
                pos_label=1,
                name=f'{model_name} (Sigmoid)',
                marker=markers['Sigmoid'],
                color=colors['Sigmoid']
            )
            
            # Isotonic calibrated if available
            if f"{model_name}_Isotonic" in self.predictions:
                y_pred_isotonic = self.predictions[f"{model_name}_Isotonic"]
                CalibrationDisplay.from_predictions(
                    self.y_test,
                    y_pred_isotonic,
                    n_bins=10,
                    strategy='uniform',
                    pos_label=1,
                    name=f'{model_name} (Isotonic)',
                    marker=markers['Isotonic'],
                    color=colors['Isotonic']
                )
        
        plt.title('Calibration Curves: All Models Comparison', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=10)
        plt.tight_layout()
        
        # Save the combined figure
        output_path = os.path.join(OUTPUT_DIR, 'calibration_curves_combined.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nCombined calibration curves saved as '{output_path}'")
        
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()
    
    def generate_summary_report(self, results_df):
        """
        Generate a comprehensive summary report
        """
        print("\n" + "=" * 70)
        print("CALIBRATION ANALYSIS SUMMARY")
        print("=" * 70)
        print()
        print(results_df.to_string(index=False))
        print()
        
        # Calculate improvements
        print("Improvements from Calibration (Brier Score Reduction):")
        print("-" * 70)
        
        for model_name in self.models_dict.keys():
            uncalibrated = results_df[
                (results_df['Model'] == model_name) & 
                (results_df['Calibration'] == 'Uncalibrated')
            ]['Brier Score'].values[0]
            
            sigmoid = results_df[
                (results_df['Model'] == model_name) & 
                (results_df['Calibration'] == 'Sigmoid (Platt)')
            ]['Brier Score'].values[0]
            
            sigmoid_improvement = ((uncalibrated - sigmoid) / uncalibrated) * 100
            print(f"{model_name} - Sigmoid: {sigmoid_improvement:.2f}% improvement")
            
            isotonic_rows = results_df[
                (results_df['Model'] == model_name) & 
                (results_df['Calibration'] == 'Isotonic')
            ]
            
            if not isotonic_rows.empty:
                isotonic = isotonic_rows['Brier Score'].values[0]
                isotonic_improvement = ((uncalibrated - isotonic) / uncalibrated) * 100
                print(f"{model_name} - Isotonic: {isotonic_improvement:.2f}% improvement")


def calibrate_and_evaluate(xgb_model, rf_model, X_train, X_test, y_train, y_test):
    """
    Main function to calibrate and evaluate classifiers
    
    Parameters:
    -----------
    xgb_model : XGBClassifier
        Trained XGBoost classifier
    rf_model : RandomForestClassifier
        Trained Random Forest classifier
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    y_train : array-like
        Training labels
    y_test : array-like
        Test labels
    
    Returns:
    --------
    results_df : DataFrame
        Calibration analysis results
    analyzer : CalibrationAnalyzer
        The analyzer instance for further analysis
    """
    models_dict = {
        'XGBoost': xgb_model,
        'RandomForest': rf_model
    }
    
    analyzer = CalibrationAnalyzer(models_dict, X_train, X_test, y_train, y_test)
    
    # Run calibration analysis
    results_df = analyzer.analyze_calibration()
    
    # Generate calibration curves
    analyzer.plot_calibration_curves()
    analyzer.plot_combined_calibration_curve()
    
    # Generate summary report
    analyzer.generate_summary_report(results_df)
    
    # Save results to CSV
    output_path = os.path.join(OUTPUT_DIR, 'calibration_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to '{output_path}'")
    
    return results_df, analyzer


if __name__ == "__main__":
    # Example usage with synthetic data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    
    print("Running calibration analysis as standalone script...")
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
    
    print("\nTraining classifiers...")
    
    # Train XGBoost
    xgb_model = XGBClassifier(
        n_estimators=100,
        random_state=42,
        verbosity=0,
        tree_method='hist'
    )
    xgb_model.fit(X_train, y_train)
    print("XGBoost trained")
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    print("RandomForest trained")
    
    # Run calibration analysis
    results_df, analyzer = calibrate_and_evaluate(
        xgb_model, rf_model, X_train, X_test, y_train, y_test
    )
