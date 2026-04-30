"""
Feature Engineering Module
Provides functions for advanced feature engineering including skewness detection,
polynomial feature generation, and recursive feature elimination.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor

# Create outputs directory if it doesn't exist
# Outputs are organized in outputs/q1/ subfolder
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'q1')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def engineer_features(X, y):
    """
    Performs comprehensive feature engineering on input features and target variable.
    
    Steps:
    1. Checks skewness of target 'y'. If skewness > 1, applies log1p transformation.
    2. Creates polynomial features (degree 2) and interaction terms using PolynomialFeatures
       without bias column.
    3. Uses Recursive Feature Elimination (RFE) with XGBoostRegressor to select the top
       15 most important features from the polynomial feature set.
    4. Returns the transformed and selected features along with the transformed target.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Input features dataframe (n_samples, n_features)
    y : pd.Series or np.ndarray
        Target variable (n_samples,)
    
    Returns:
    --------
    X_selected : pd.DataFrame
        Transformed and selected features with top 15 features (n_samples, 15)
    y_transformed : np.ndarray
        Transformed target variable. Log-transformed if original skewness > 1,
        otherwise unchanged (n_samples,)
    
    Raises:
    -------
    ValueError
        If X is not a DataFrame, y is not a Series/array, or if number of samples
        don't match between X and y
    
    Example:
    --------
    >>> from sklearn.datasets import fetch_california_housing
    >>> import pandas as pd
    >>> housing = fetch_california_housing()
    >>> X = pd.DataFrame(housing.data, columns=housing.feature_names)
    >>> y = pd.Series(housing.target, name='Price')
    >>> X_selected, y_transformed = engineer_features(X, y)
    >>> print(X_selected.shape)  # (20640, 15)
    >>> print(y_transformed.shape)  # (20640,)
    """
    
    # ========================================================================
    # INPUT VALIDATION
    # ========================================================================
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")
    
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise ValueError("y must be a pandas Series or numpy array")
    
    if len(X) != len(y):
        raise ValueError(f"X and y must have same number of samples. "
                        f"Got X: {len(X)}, y: {len(y)}")
    
    # Convert y to numpy array if it's a Series
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = np.array(y)
    
    print("="*70)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*70)
    
    # ========================================================================
    # STEP 1: CHECK SKEWNESS AND TRANSFORM TARGET IF NEEDED
    # ========================================================================
    print("\nStep 1: Analyzing target variable skewness...")
    print("-"*70)
    
    skewness = stats.skew(y_array)
    print(f"Skewness of target variable y: {skewness:.4f}")
    
    if skewness > 1:
        print(f"Skewness > 1 detected! Applying log1p transformation...")
        y_transformed = np.log1p(y_array)
        print(f"Transformed target skewness: {stats.skew(y_transformed):.4f}")
        transformation_applied = True
    else:
        print(f"Skewness ≤ 1. No transformation applied.")
        y_transformed = y_array.copy()
        transformation_applied = False
    
    # ========================================================================
    # STEP 2: CREATE POLYNOMIAL FEATURES
    # ========================================================================
    print("\nStep 2: Creating polynomial features (degree 2)...")
    print("-"*70)
    
    print(f"Original number of features: {X.shape[1]}")
    
    # Create polynomial features without bias term
    poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_transformer.fit_transform(X)
    
    # Get feature names for polynomial features
    feature_names = poly_transformer.get_feature_names_out(X.columns)
    X_poly_df = pd.DataFrame(X_poly, columns=feature_names)
    
    print(f"Number of polynomial features (degree 2, no bias): {X_poly_df.shape[1]}")
    print(f"Feature examples: {list(feature_names[:5])}...")
    
    # ========================================================================
    # STEP 3: RECURSIVE FEATURE ELIMINATION (RFE)
    # ========================================================================
    print("\nStep 3: Performing Recursive Feature Elimination (RFE)...")
    print("-"*70)
    
    # Initialize XGBoost regressor as base estimator
    print("Initializing XGBoostRegressor as base estimator...")
    xgb_estimator = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    
    # Apply RFE to select top 15 features
    print("Running RFE to select top 15 features...")
    rfe = RFE(estimator=xgb_estimator, n_features_to_select=15, step=10)
    rfe.fit(X_poly_df, y_transformed)
    
    # Get selected features
    selected_feature_mask = rfe.support_
    selected_features = feature_names[selected_feature_mask]
    X_selected = X_poly_df.loc[:, selected_features]
    
    print(f"Number of features selected: {X_selected.shape[1]}")
    print(f"\nSelected features:")
    for i, feature in enumerate(selected_features, 1):
        print(f"  {i:2d}. {feature}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*70)
    print(f"Original feature count: {X.shape[1]}")
    print(f"Polynomial features created: {X_poly_df.shape[1]}")
    print(f"Features selected by RFE: {X_selected.shape[1]}")
    print(f"Target transformation applied: {'Yes (log1p)' if transformation_applied else 'No'}")
    print(f"Original target skewness: {skewness:.4f}")
    print(f"Final target skewness: {stats.skew(y_transformed):.4f}")
    print(f"\nFinal dataset shape: X={X_selected.shape}, y={y_transformed.shape}")
    print("="*70 + "\n")
    
    return X_selected, y_transformed


def get_feature_importance(X_selected, y_transformed, estimator=None):
    """
    Train a model on the selected features and return feature importances.
    
    Parameters:
    -----------
    X_selected : pd.DataFrame
        Selected features from engineer_features function
    y_transformed : np.ndarray
        Transformed target from engineer_features function
    estimator : estimator object, optional
        Sklearn-compatible estimator. If None, uses XGBRegressor.
    
    Returns:
    --------
    importance_df : pd.DataFrame
        DataFrame with feature names and their importance scores, sorted by importance
    """
    
    if estimator is None:
        estimator = XGBRegressor(n_estimators=100, max_depth=5, 
                                learning_rate=0.1, random_state=42, verbosity=0)
    
    # Train the model
    estimator.fit(X_selected, y_transformed)
    
    # Get feature importances
    importances = estimator.feature_importances_
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'Feature': X_selected.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    
    return importance_df
