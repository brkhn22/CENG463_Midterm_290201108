"""
Residual Diagnostics and Robust Regression
Provides functions for comprehensive residual analysis and demonstrates
robust regression using HuberRegressor.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create outputs directory if it doesn't exist
# Outputs are organized in outputs/q1/ subfolder
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'q1')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def residual_diagnostics(y_true, y_pred, model_name="Model", figsize=(14, 5)):
    """
    Perform comprehensive residual diagnostics for a fitted regression model.
    
    Creates a 1x2 figure showing:
    1. Fitted Values vs. Residuals plot to check homoscedasticity
    2. Q-Q plot to check normality of residuals
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str, default="Model"
        Name of the model for the title
    figsize : tuple, default=(14, 5)
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object
    axes : array of matplotlib.axes.Axes
        The axes objects
    residuals : np.ndarray
        Calculated residuals (y_true - y_pred)
    
    Notes:
    ------
    - Homoscedasticity: Look for random scatter around the horizontal line at 0.
      If you see a funnel pattern (increasing/decreasing spread), there may be
      heteroscedasticity.
    - Normality: Points should closely follow the diagonal line in the Q-Q plot.
      Deviations at the tails indicate departure from normality.
    """
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    print("="*70)
    print("RESIDUAL DIAGNOSTICS")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Sample size: {len(y_true)}")
    print(f"\nResidual Statistics:")
    print(f"  Mean: {np.mean(residuals):.6f}")
    print(f"  Std Dev: {np.std(residuals):.6f}")
    print(f"  Min: {np.min(residuals):.6f}")
    print(f"  Max: {np.max(residuals):.6f}")
    print(f"  Skewness: {stats.skew(residuals):.6f}")
    print(f"  Kurtosis: {stats.kurtosis(residuals):.6f}")
    
    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'Residual Diagnostics - {model_name}', fontsize=14, fontweight='bold')
    
    # ========================================================================
    # LEFT PLOT: FITTED VALUES vs RESIDUALS (Homoscedasticity Check)
    # ========================================================================
    ax1 = axes[0]
    
    ax1.scatter(y_pred, residuals, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Residual Line')
    
    # Add LOWESS (locally weighted regression) line to show trend
    from scipy.signal import savgol_filter
    sorted_indices = np.argsort(y_pred)
    y_pred_sorted = y_pred[sorted_indices]
    residuals_sorted = residuals[sorted_indices]
    
    # Apply simple smoothing to show any trend
    if len(y_pred_sorted) > 10:
        window_length = min(51, len(y_pred_sorted) - 1 if len(y_pred_sorted) % 2 == 0 else len(y_pred_sorted))
        if window_length % 2 == 0:
            window_length -= 1
        if window_length >= 5:
            try:
                smoothed = savgol_filter(residuals_sorted, window_length=window_length, polyorder=3)
                ax1.plot(y_pred_sorted, smoothed, 'b-', linewidth=2, alpha=0.7, label='Trend Line')
            except:
                pass
    
    ax1.set_xlabel('Fitted Values', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Residuals', fontsize=11, fontweight='bold')
    ax1.set_title('Fitted Values vs. Residuals\n(Check for Homoscedasticity)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # ========================================================================
    # RIGHT PLOT: Q-Q PLOT (Normality Check)
    # ========================================================================
    ax2 = axes[1]
    
    # Use statsmodels for Q-Q plot
    sm.qqplot(residuals, line='45', ax=ax2, markersize=6, markerfacecolor='lightblue',
              markeredgecolor='black', markeredgewidth=0.5)
    
    ax2.set_title('Q-Q Plot\n(Check for Normality)', fontsize=12)
    ax2.set_xlabel('Theoretical Quantiles', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Sample Quantiles', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # ========================================================================
    # INTERPRETATION GUIDE
    # ========================================================================
    print("\n" + "-"*70)
    print("INTERPRETATION GUIDE:")
    print("-"*70)
    print("\n1. HOMOSCEDASTICITY (Left Plot - Fitted vs Residuals):")
    print("   ✓ Good: Random scatter around the red line with no pattern")
    print("   ✗ Bad: Funnel shape (increasing/decreasing spread) or curved pattern")
    print(f"   Range of residuals: [{np.min(residuals):.4f}, {np.max(residuals):.4f}]")
    
    print("\n2. NORMALITY (Right Plot - Q-Q Plot):")
    print("   ✓ Good: Points closely follow the diagonal line")
    print("   ✗ Bad: Systematic deviation from the diagonal (S-shaped curve)")
    print("   ✗ Bad: Heavy tails (points deviate significantly at extremes)")
    
    # Perform Shapiro-Wilk test for normality
    if len(residuals) <= 5000:  # Shapiro-Wilk works best for n <= 5000
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        print(f"\nShapiro-Wilk Test for Normality:")
        print(f"   Test Statistic: {shapiro_stat:.6f}")
        print(f"   p-value: {shapiro_p:.6f}")
        if shapiro_p > 0.05:
            print(f"   ✓ Residuals appear NORMAL (p > 0.05)")
        else:
            print(f"   ✗ Residuals may NOT be normal (p < 0.05)")
    
    print("="*70 + "\n")
    
    return fig, axes, residuals


def train_and_evaluate_huber(X_train, X_test, y_train, y_test, epsilon=1.35):
    """
    Train and evaluate both LinearRegression and HuberRegressor,
    comparing their robustness to outliers.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    X_test : pd.DataFrame or np.ndarray
        Test features
    y_train : pd.Series or np.ndarray
        Training target
    y_test : pd.Series or np.ndarray
        Test target
    epsilon : float, default=1.35
        The parameter in Huber regressor that controls the number of samples
        should be classified as outliers. Smaller epsilon = more robustness.
    
    Returns:
    --------
    results : dict
        Dictionary containing model artifacts and metrics
    """
    
    print("="*70)
    print("ROBUST REGRESSION: LINEAR REGRESSION vs HUBER REGRESSOR")
    print("="*70)
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # ========================================================================
    # MODEL 1: STANDARD LINEAR REGRESSION (OLS)
    # ========================================================================
    print("\n" + "-"*70)
    print("1. STANDARD LINEAR REGRESSION (OLS)")
    print("-"*70)
    
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    
    rmse_lr = calculate_rmse(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    
    print(f"RMSE (Test Set): {rmse_lr:.6f}")
    print(f"R² Score (Test Set): {r2_lr:.6f}")
    print(f"\nCharacteristics:")
    print("  • Assumes normally distributed errors")
    print("  • Minimizes sum of squared residuals")
    print("  • Highly sensitive to outliers (outliers have large influence)")
    print("  • May have inflated predictions if outliers are present")
    
    # ========================================================================
    # MODEL 2: HUBER REGRESSOR (ROBUST)
    # ========================================================================
    print("\n" + "-"*70)
    print("2. HUBER REGRESSOR (ROBUST)")
    print("-"*70)
    
    huber_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', HuberRegressor(epsilon=epsilon, max_iter=1000))
    ])
    
    huber_pipeline.fit(X_train, y_train)
    y_pred_huber = huber_pipeline.predict(X_test)
    
    rmse_huber = calculate_rmse(y_test, y_pred_huber)
    r2_huber = r2_score(y_test, y_pred_huber)
    
    print(f"RMSE (Test Set): {rmse_huber:.6f}")
    print(f"R² Score (Test Set): {r2_huber:.6f}")
    print(f"Epsilon (outlier threshold): {epsilon}")
    print(f"\nCharacteristics:")
    print("  • Combines least squares and absolute error loss")
    print("  • Uses absolute loss for large residuals (near outliers)")
    print("  • Uses squared loss for small residuals (inliers)")
    print("  • Robust to outliers while maintaining efficiency")
    print("  • Less sensitive to extreme values")
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    comparison = pd.DataFrame({
        'Metric': ['RMSE', 'R²'],
        'LinearRegression': [rmse_lr, r2_lr],
        'HuberRegressor': [rmse_huber, r2_huber],
        'Difference': [rmse_lr - rmse_huber, r2_huber - r2_lr]
    })
    
    print(f"\n{comparison.to_string(index=False)}\n")
    
    # Determine which is better
    print("Summary:")
    if rmse_huber < rmse_lr:
        improvement = ((rmse_lr - rmse_huber) / rmse_lr) * 100
        print(f"  ✓ HuberRegressor has LOWER RMSE by {improvement:.2f}%")
        print(f"    → Less sensitive to outliers in the test set")
    else:
        difference = ((rmse_lr - rmse_huber) / rmse_huber) * 100
        print(f"  • LinearRegression has lower RMSE by {abs(difference):.2f}%")
        print(f"    → Test set may have few outliers or is well-behaved")
    
    if r2_huber > r2_lr:
        print(f"  ✓ HuberRegressor has HIGHER R² ")
        print(f"    → Better fit to the data overall")
    else:
        print(f"  • LinearRegression has higher R²")
        print(f"    → May indicate less outlier influence in test set")
    
    print("\nRecommendation:")
    print("  • Use HuberRegressor when dealing with datasets containing outliers")
    print("  • Use LinearRegression for clean, normally distributed data")
    print("  • HuberRegressor provides better generalization in noisy environments")
    
    print("="*70 + "\n")
    
    # Store results
    results = {
        'lr_pipeline': lr_pipeline,
        'huber_pipeline': huber_pipeline,
        'y_pred_lr': y_pred_lr,
        'y_pred_huber': y_pred_huber,
        'rmse_lr': rmse_lr,
        'r2_lr': r2_lr,
        'rmse_huber': rmse_huber,
        'r2_huber': r2_huber,
        'comparison': comparison
    }
    
    return results
