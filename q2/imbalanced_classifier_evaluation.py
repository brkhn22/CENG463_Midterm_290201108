"""
Question 2: Evaluating Multiple Classifiers on Imbalanced Dataset
Uses imblearn.pipeline for proper handling of resampling strategies
Implements cost-sensitive learning and various sampling techniques
"""

import os
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, 
    auc, precision_recall_curve, matthews_corrcoef, balanced_accuracy_score
)
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb

warnings.filterwarnings('ignore')

# Outputs are organized in outputs/q2/ subfolder
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'q2')
os.makedirs(OUTPUT_DIR, exist_ok=True)


class ImbalancedClassifierEvaluator:
    """
    Comprehensive evaluator for multiple classifiers on imbalanced datasets
    """
    
    def __init__(self, X, y, n_splits=5, random_state=42):
        """
        Initialize the evaluator
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        y : array-like of shape (n_samples,)
            Target variable
        n_splits : int, default=5
            Number of cross-validation folds
        random_state : int, default=42
            Random state for reproducibility
        """
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.results = []
        
        # Calculate class weights and frequencies for cost-sensitive learning
        self._calculate_class_weights()
    
    def _calculate_class_weights(self):
        """Calculate class weights for cost-sensitive learning"""
        classes, counts = np.unique(self.y, return_counts=True)
        self.class_counts = dict(zip(classes, counts))
        self.class_weights = {}
        total = len(self.y)
        
        for cls, count in self.class_counts.items():
            self.class_weights[cls] = total / (len(self.class_counts) * count)
        
        # For XGBoost scale_pos_weight
        self.scale_pos_weight = self.class_counts.get(0, 1) / self.class_counts.get(1, 1)
        
        print(f"Class distribution: {self.class_counts}")
        print(f"Scale pos weight (for XGBoost): {self.scale_pos_weight:.4f}")
        print(f"Class weights: {self.class_weights}\n")
    
    def _create_scoring_dict(self):
        """
        Create a dictionary of scoring functions for cross-validation
        """
        scoring = {
            'precision': 'precision',
            'recall': 'recall',
            'f1_macro': 'f1_macro',
            'f1_micro': 'f1_micro',
            'roc_auc': 'roc_auc',
            'balanced_accuracy': 'balanced_accuracy'
        }
        return scoring
    
    def _calculate_pr_auc_and_mcc(self):
        """
        Calculate PR-AUC and MCC using custom cross-validation
        """
        pr_auc_scores = []
        mcc_scores = []
        
        for train_idx, test_idx in self.cv.split(self.X, self.y):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            # For this calculation, we'll use a simple model
            # In practice, this would be calculated for each pipeline separately
            # Here we demonstrate the approach
            pass
        
        return pr_auc_scores, mcc_scores
    
    def _custom_scoring(self, pipeline_name, pipeline):
        """
        Perform custom cross-validation with all metrics including PR-AUC and MCC
        """
        fold_results = {
            'precision': [], 'recall': [], 'f1_macro': [], 'f1_micro': [],
            'roc_auc': [], 'pr_auc': [], 'mcc': [], 'balanced_accuracy': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(self.cv.split(self.X, self.y)):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            # Fit the pipeline
            pipeline.fit(X_train, y_train)
            
            # Get predictions
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            fold_results['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            fold_results['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            fold_results['f1_macro'].append(f1_score(y_test, y_pred, average='macro', zero_division=0))
            fold_results['f1_micro'].append(f1_score(y_test, y_pred, average='micro', zero_division=0))
            fold_results['roc_auc'].append(roc_auc_score(y_test, y_pred_proba))
            fold_results['balanced_accuracy'].append(balanced_accuracy_score(y_test, y_pred))
            fold_results['mcc'].append(matthews_corrcoef(y_test, y_pred))
            
            # Calculate PR-AUC
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall_vals, precision_vals)
            fold_results['pr_auc'].append(pr_auc)
        
        return fold_results
    
    def _create_smote_pipeline(self, classifier, classifier_name):
        """Create pipeline with SMOTE resampling"""
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=self.random_state, k_neighbors=5)),
            ('classifier', classifier)
        ])
        return pipeline, f"{classifier_name}_SMOTE"
    
    def _create_adasyn_pipeline(self, classifier, classifier_name):
        """Create pipeline with ADASYN resampling"""
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('adasyn', ADASYN(random_state=self.random_state, n_neighbors=5)),
            ('classifier', classifier)
        ])
        return pipeline, f"{classifier_name}_ADASYN"
    
    def _create_undersampler_pipeline(self, classifier, classifier_name):
        """Create pipeline with RandomUnderSampler resampling"""
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('undersampler', RandomUnderSampler(random_state=self.random_state)),
            ('classifier', classifier)
        ])
        return pipeline, f"{classifier_name}_RUS"
    
    def _create_cost_sensitive_pipeline(self, classifier, classifier_name):
        """Create pipeline for cost-sensitive classifier (XGBoost, MLP)"""
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier)
        ])
        return pipeline, classifier_name
    
    def evaluate_logistic_regression(self):
        """Evaluate LogisticRegression with different resampling strategies"""
        print("Evaluating LogisticRegression...")
        print("-" * 60)
        
        base_lr = LogisticRegression(
            max_iter=1000, 
            random_state=self.random_state,
            class_weight='balanced'
        )
        
        # SMOTE
        pipeline, name = self._create_smote_pipeline(base_lr, "LogisticRegression")
        fold_results = self._custom_scoring(name, pipeline)
        mean_results = {k: np.mean(v) for k, v in fold_results.items()}
        mean_results['Model'] = name
        self.results.append(mean_results)
        print(f"{name}: F1-Macro={mean_results['f1_macro']:.4f}, ROC-AUC={mean_results['roc_auc']:.4f}")
        
        # ADASYN
        pipeline, name = self._create_adasyn_pipeline(base_lr, "LogisticRegression")
        fold_results = self._custom_scoring(name, pipeline)
        mean_results = {k: np.mean(v) for k, v in fold_results.items()}
        mean_results['Model'] = name
        self.results.append(mean_results)
        print(f"{name}: F1-Macro={mean_results['f1_macro']:.4f}, ROC-AUC={mean_results['roc_auc']:.4f}")
        
        # RandomUnderSampler
        pipeline, name = self._create_undersampler_pipeline(base_lr, "LogisticRegression")
        fold_results = self._custom_scoring(name, pipeline)
        mean_results = {k: np.mean(v) for k, v in fold_results.items()}
        mean_results['Model'] = name
        self.results.append(mean_results)
        print(f"{name}: F1-Macro={mean_results['f1_macro']:.4f}, ROC-AUC={mean_results['roc_auc']:.4f}\n")
    
    def evaluate_random_forest(self):
        """Evaluate RandomForestClassifier with different resampling strategies"""
        print("Evaluating RandomForestClassifier...")
        print("-" * 60)
        
        base_rf = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # SMOTE
        pipeline, name = self._create_smote_pipeline(base_rf, "RandomForest")
        fold_results = self._custom_scoring(name, pipeline)
        mean_results = {k: np.mean(v) for k, v in fold_results.items()}
        mean_results['Model'] = name
        self.results.append(mean_results)
        print(f"{name}: F1-Macro={mean_results['f1_macro']:.4f}, ROC-AUC={mean_results['roc_auc']:.4f}")
        
        # ADASYN
        pipeline, name = self._create_adasyn_pipeline(base_rf, "RandomForest")
        fold_results = self._custom_scoring(name, pipeline)
        mean_results = {k: np.mean(v) for k, v in fold_results.items()}
        mean_results['Model'] = name
        self.results.append(mean_results)
        print(f"{name}: F1-Macro={mean_results['f1_macro']:.4f}, ROC-AUC={mean_results['roc_auc']:.4f}")
        
        # RandomUnderSampler
        pipeline, name = self._create_undersampler_pipeline(base_rf, "RandomForest")
        fold_results = self._custom_scoring(name, pipeline)
        mean_results = {k: np.mean(v) for k, v in fold_results.items()}
        mean_results['Model'] = name
        self.results.append(mean_results)
        print(f"{name}: F1-Macro={mean_results['f1_macro']:.4f}, ROC-AUC={mean_results['roc_auc']:.4f}\n")
    
    def evaluate_xgboost(self):
        """Evaluate XGBClassifier with scale_pos_weight for cost-sensitive learning"""
        print("Evaluating XGBClassifier (Cost-Sensitive)...")
        print("-" * 60)
        
        xgb_classifier = XGBClassifier(
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            tree_method='hist',
            verbosity=0
        )
        
        pipeline, name = self._create_cost_sensitive_pipeline(xgb_classifier, "XGBoost_CostSensitive")
        fold_results = self._custom_scoring(name, pipeline)
        mean_results = {k: np.mean(v) for k, v in fold_results.items()}
        mean_results['Model'] = name
        self.results.append(mean_results)
        print(f"{name}: F1-Macro={mean_results['f1_macro']:.4f}, ROC-AUC={mean_results['roc_auc']:.4f}\n")
    
    def evaluate_mlp(self):
        """Evaluate MLPClassifier with sample_weights for cost-sensitive learning"""
        print("Evaluating MLPClassifier (Cost-Sensitive with Sample Weights)...")
        print("-" * 60)
        
        mlp_classifier = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        pipeline, name = self._create_cost_sensitive_pipeline(mlp_classifier, "MLP_CostSensitive")
        
        # Custom cross-validation with sample_weights
        fold_results = {
            'precision': [], 'recall': [], 'f1_macro': [], 'f1_micro': [],
            'roc_auc': [], 'pr_auc': [], 'mcc': [], 'balanced_accuracy': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(self.cv.split(self.X, self.y)):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            # Generate sample weights based on class weights
            sample_weights = np.array([self.class_weights[y] for y in y_train])
            
            # Create pipeline without classifier for preprocessing
            preprocessing = Pipeline([('scaler', StandardScaler())])
            X_train_scaled = preprocessing.fit_transform(X_train)
            X_test_scaled = preprocessing.transform(X_test)
            
            # Fit MLP with sample_weights
            mlp_classifier.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            
            # Get predictions
            y_pred = mlp_classifier.predict(X_test_scaled)
            y_pred_proba = mlp_classifier.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            fold_results['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            fold_results['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            fold_results['f1_macro'].append(f1_score(y_test, y_pred, average='macro', zero_division=0))
            fold_results['f1_micro'].append(f1_score(y_test, y_pred, average='micro', zero_division=0))
            fold_results['roc_auc'].append(roc_auc_score(y_test, y_pred_proba))
            fold_results['balanced_accuracy'].append(balanced_accuracy_score(y_test, y_pred))
            fold_results['mcc'].append(matthews_corrcoef(y_test, y_pred))
            
            # Calculate PR-AUC
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall_vals, precision_vals)
            fold_results['pr_auc'].append(pr_auc)
        
        mean_results = {k: np.mean(v) for k, v in fold_results.items()}
        mean_results['Model'] = name
        self.results.append(mean_results)
        print(f"{name}: F1-Macro={mean_results['f1_macro']:.4f}, ROC-AUC={mean_results['roc_auc']:.4f}\n")
    
    def evaluate_all(self):
        """Run evaluation for all classifiers"""
        print("=" * 60)
        print("EVALUATING ALL CLASSIFIERS ON IMBALANCED DATASET")
        print("=" * 60)
        print()
        
        self.evaluate_logistic_regression()
        self.evaluate_random_forest()
        self.evaluate_xgboost()
        self.evaluate_mlp()
    
    def get_results_dataframe(self):
        """Return results as a DataFrame"""
        if not self.results:
            raise ValueError("No results available. Run evaluate_all() first.")
        
        results_df = pd.DataFrame(self.results)
        
        # Reorder columns for better readability
        metric_cols = ['precision', 'recall', 'f1_macro', 'f1_micro', 
                      'roc_auc', 'pr_auc', 'mcc', 'balanced_accuracy']
        cols_order = ['Model'] + [col for col in metric_cols if col in results_df.columns]
        results_df = results_df[cols_order]
        
        return results_df


def main(X, y):
    """
    Main function to evaluate classifiers
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Target variable
    
    Returns:
    --------
    results_df : DataFrame
        Evaluation results with mean scores across folds
    """
    evaluator = ImbalancedClassifierEvaluator(X, y, n_splits=5, random_state=42)
    evaluator.evaluate_all()
    results_df = evaluator.get_results_dataframe()
    
    print("=" * 80)
    print("EVALUATION RESULTS (Mean Scores Across 5 Folds)")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print()
    
    return results_df


if __name__ == "__main__":
    # Example usage with synthetic data
    from sklearn.datasets import make_classification
    
    print("Running classifier evaluation as standalone script...")
    
    # Create imbalanced dataset for demonstration
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        weights=[0.95, 0.05],  # 95% majority, 5% minority
        random_state=42
    )
    
    # Run evaluation
    results_df = main(X, y)
    
    # Save results to CSV
    output_path = os.path.join(OUTPUT_DIR, 'classifier_evaluation_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to '{output_path}'")
