"""
Robust Model Evaluation Loop using Scikit-learn
Implements RepeatedKFold cross-validation with multiple regression models,
comprehensive metrics calculation, and statistical significance testing.
"""

import os
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict

from sklearn.model_selection import RepeatedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from xgboost import XGBRegressor
from scipy import stats

warnings.filterwarnings('ignore')

# Create outputs directory if it doesn't exist
# Outputs are organized in outputs/q1/ subfolder
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'q1')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_adjusted_r2(r2, n_samples, n_features):
    """
    Calculate adjusted R² score.
    
    Parameters:
    -----------
    r2 : float
        R² score
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    
    Returns:
    --------
    adjusted_r2 : float
        Adjusted R² score
    """
    if n_samples - n_features - 1 <= 0:
        return r2
    
    adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    return adjusted_r2


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_models(X, y, models_dict=None, cv=None, random_state=42):
    """
    Comprehensive model evaluation using RepeatedKFold cross-validation.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Feature matrix (n_samples, n_features)
    y : pd.Series or np.ndarray
        Target variable (n_samples,)
    models_dict : dict, optional
        Dictionary of models to evaluate. If None, uses default models.
    cv : RepeatedKFold, optional
        Cross-validation strategy. If None, creates RepeatedKFold(5, 3)
    random_state : int, default=42
        Random state for reproducibility
    
    Returns:
    --------
    results_df : pd.DataFrame
        Results summary with mean and std for each metric per model
    all_metrics : dict
        Dictionary containing all fold metrics for each model
    cv_object : RepeatedKFold
        The cross-validation object used
    """
    
    # Set default CV strategy
    if cv is None:
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=random_state)
    
    # Set default models if not provided
    if models_dict is None:
        models_dict = get_default_models(random_state)
    
    print("="*80)
    print("ROBUST MODEL EVALUATION PIPELINE")
    print("="*80)
    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Cross-validation strategy: {cv}")
    print(f"Models to evaluate: {list(models_dict.keys())}\n")
    
    # Get total number of folds
    n_folds = cv.get_n_splits()
    print(f"Total evaluation folds: {n_folds}\n")
    
    # Storage for metrics
    all_metrics = defaultdict(lambda: defaultdict(list))
    model_pipelines = {}
    
    # ========================================================================
    # ITERATE THROUGH CROSS-VALIDATION FOLDS
    # ========================================================================
    fold_count = 0
    
    for train_idx, test_idx in cv.split(X):
        fold_count += 1
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        print(f"Fold {fold_count}/{n_folds}: Train={len(X_train)}, Test={len(X_test)}")
        
        # ====================================================================
        # EVALUATE EACH MODEL
        # ====================================================================
        for model_name, model in models_dict.items():
            
            # Create pipeline with StandardScaler
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            # Fit pipeline
            pipeline.fit(X_train, y_train)
            
            # Store pipeline (from first fold for later use)
            if fold_count == 1:
                model_pipelines[model_name] = pipeline
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            rmse = calculate_rmse(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            adjusted_r2 = calculate_adjusted_r2(r2, len(y_test), X_test.shape[1])
            mape = mean_absolute_percentage_error(y_test, y_pred)
            evs = explained_variance_score(y_test, y_pred)
            
            # Store metrics
            all_metrics[model_name]['RMSE'].append(rmse)
            all_metrics[model_name]['MAE'].append(mae)
            all_metrics[model_name]['R²'].append(r2)
            all_metrics[model_name]['Adjusted R²'].append(adjusted_r2)
            all_metrics[model_name]['MAPE'].append(mape)
            all_metrics[model_name]['EVS'].append(evs)
    
    print(f"\n✓ All {n_folds} folds evaluated successfully!\n")
    
    # ========================================================================
    # CREATE SUMMARY TABLE
    # ========================================================================
    print("="*80)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("="*80 + "\n")
    
    summary_data = []
    metric_names = ['RMSE', 'MAE', 'R²', 'Adjusted R²', 'MAPE', 'EVS']
    
    for model_name in models_dict.keys():
        for metric in metric_names:
            metric_values = all_metrics[model_name][metric]
            mean_val = np.mean(metric_values)
            std_val = np.std(metric_values)
            
            summary_data.append({
                'Model': model_name,
                'Metric': metric,
                'Mean': mean_val,
                'Std Dev': std_val,
                'Format': f"{mean_val:.4f} ± {std_val:.4f}"
            })
    
    results_df = pd.DataFrame(summary_data)
    
    # Print summary by model
    for model_name in models_dict.keys():
        print(f"\n{model_name}")
        print("-" * 80)
        model_results = results_df[results_df['Model'] == model_name][
            ['Metric', 'Mean', 'Std Dev', 'Format']
        ].reset_index(drop=True)
        print(model_results.to_string(index=False))
    
    print("\n" + "="*80)
    
    return results_df, all_metrics, cv


def get_default_models(random_state=42):
    """
    Create dictionary of default models for evaluation.
    
    Parameters:
    -----------
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    models : dict
        Dictionary of sklearn models
    """
    
    print("Initializing models...")
    
    # LinearRegression
    lr = LinearRegression()
    
    # RidgeCV
    ridge = RidgeCV(alphas=np.logspace(-2, 2, 20), cv=5)
    
    # LassoCV
    lasso = LassoCV(alphas=np.logspace(-4, 1, 20), cv=5, random_state=random_state)
    
    # ElasticNetCV with l1_ratios
    elasticnet = ElasticNetCV(
        l1_ratio=[.1, .5, .7, .9, .95, 1],
        alphas=np.logspace(-4, 1, 20),
        cv=5,
        random_state=random_state
    )
    
    # XGBRegressor with RandomizedSearchCV for hyperparameter tuning
    print("Setting up XGBRegressor with RandomizedSearchCV...")
    xgb_base = XGBRegressor(random_state=random_state, verbosity=0, n_jobs=-1)
    
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2]
    }
    
    xgb_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_grid,
        n_iter=10,
        cv=5,
        random_state=random_state,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
    
    models = {
        'LinearRegression': lr,
        'RidgeCV': ridge,
        'LassoCV': lasso,
        'ElasticNetCV': elasticnet,
        'XGBRegressor (Tuned)': xgb_search
    }
    
    print("✓ Models initialized\n")
    
    return models


def perform_statistical_test(all_metrics, model1='XGBRegressor (Tuned)', 
                             model2='LinearRegression', alpha=0.05):
    """
    Perform paired t-test between two models using R² scores.
    
    Parameters:
    -----------
    all_metrics : dict
        Metrics dictionary from evaluate_models()
    model1 : str
        First model name
    model2 : str
        Second model name
    alpha : float
        Significance level
    
    Returns:
    --------
    ttest_result : dict
        Dictionary containing t-statistic, p-value, and interpretation
    """
    
    print("="*80)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("="*80 + "\n")
    
    # Extract R² scores
    r2_model1 = np.array(all_metrics[model1]['R²'])
    r2_model2 = np.array(all_metrics[model2]['R²'])
    
    print(f"Paired t-test: {model1} vs {model2}")
    print(f"R² scores - {model1}: {r2_model1}")
    print(f"R² scores - {model2}: {r2_model2}\n")
    
    # Perform paired t-test
    t_statistic, p_value = stats.ttest_rel(r2_model1, r2_model2)
    
    # Calculate mean differences
    mean_diff = np.mean(r2_model1 - r2_model2)
    std_diff = np.std(r2_model1 - r2_model2)
    
    # Determine significance
    is_significant = p_value < alpha
    
    print(f"Sample size: {len(r2_model1)}")
    print(f"Mean R² - {model1}: {np.mean(r2_model1):.4f}")
    print(f"Mean R² - {model2}: {np.mean(r2_model2):.4f}")
    print(f"Mean difference: {mean_diff:.4f}")
    print(f"Std of differences: {std_diff:.4f}\n")
    
    print(f"t-statistic: {t_statistic:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Significance level (α): {alpha}")
    print(f"Result: {'SIGNIFICANT' if is_significant else 'NOT SIGNIFICANT'}")
    
    if is_significant:
        if mean_diff > 0:
            print(f"\n✓ {model1} performs SIGNIFICANTLY BETTER than {model2} (p < {alpha})")
        else:
            print(f"\n✓ {model2} performs SIGNIFICANTLY BETTER than {model1} (p < {alpha})")
    else:
        print(f"\n✗ No statistically significant difference between models (p ≥ {alpha})")
    
    print("="*80 + "\n")
    
    return {
        't_statistic': t_statistic,
        'p_value': p_value,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'is_significant': is_significant,
        'alpha': alpha
    }
