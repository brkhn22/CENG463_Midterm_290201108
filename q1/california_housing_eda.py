"""
Exploratory Data Analysis for California Housing Dataset
This script loads the California Housing dataset and performs comprehensive EDA
including distributions, correlations, pairplots, and outlier detection.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Create outputs directory if it doesn't exist
# Outputs are organized in outputs/q1/ subfolder
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'q1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the California Housing dataset
print("Loading California Housing dataset...")
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='Price')

# Combine features and target into single dataframe
df = pd.concat([X, y], axis=1)

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nDataset info:\n{df.info()}")
print(f"\nBasic statistics:\n{df.describe()}")

# ============================================================================
# 1. PLOT HISTOGRAMS FOR ALL FEATURE DISTRIBUTIONS
# ============================================================================
print("\n" + "="*70)
print("1. Plotting histograms for all feature distributions...")
print("="*70)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

for idx, column in enumerate(df.columns):
    axes[idx].hist(df[column], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Distribution of {column}', fontsize=10, fontweight='bold')
    axes[idx].set_xlabel(column)
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, 'histograms_all_features.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Histograms saved as '{output_path}'")
plt.close()

# ============================================================================
# 2. PLOT CORRELATION HEATMAP
# ============================================================================
print("\n" + "="*70)
print("2. Plotting correlation heatmap...")
print("="*70)

plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.2f')
plt.title('Correlation Heatmap - All Variables', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Correlation heatmap saved as '{output_path}'")
plt.close()

# ============================================================================
# 3. IDENTIFY TOP 3 FEATURES CORRELATED WITH TARGET & PAIRPLOT
# ============================================================================
print("\n" + "="*70)
print("3. Identifying top 3 features correlated with target...")
print("="*70)

# Get correlations with target (excluding the target itself)
target_correlations = correlation_matrix['Price'].drop('Price').abs().sort_values(ascending=False)
print(f"\nCorrelations with Price (sorted by absolute value):\n{target_correlations}")

top_3_features = target_correlations.head(3).index.tolist()
print(f"\nTop 3 features most correlated with Price:\n{top_3_features}")

# Create pairplot for top 3 features with target
pairplot_data = df[top_3_features + ['Price']]
print(f"\nCreating pairplot for: {top_3_features + ['Price']}")

plt.figure(figsize=(14, 12))
pairplot = sns.pairplot(pairplot_data, diag_kind='hist', plot_kws={'alpha': 0.6, 's': 30},
                        diag_kws={'bins': 30, 'edgecolor': 'black'})
pairplot.fig.suptitle('Pairplot of Top 3 Features + Target Variable', 
                       fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, 'pairplot_top3_features.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Pairplot saved as '{output_path}'")
plt.close()

# ============================================================================
# 4. OUTLIER DETECTION USING IQR METHOD
# ============================================================================
print("\n" + "="*70)
print("4. Identifying outliers using IQR method...")
print("="*70)

def detect_outliers_iqr(dataframe):
    """
    Identifies and counts outliers for each continuous feature using IQR method.
    
    Parameters:
    -----------
    dataframe : pd.DataFrame
        Input dataframe with continuous features
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with columns: Feature, Q1, Q3, IQR, Lower_Bound, Upper_Bound, 
        Outlier_Count, Outlier_Percentage
    """
    
    results = []
    
    for column in dataframe.columns:
        # Calculate Q1, Q3, and IQR
        Q1 = dataframe[column].quantile(0.25)
        Q3 = dataframe[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Calculate bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = dataframe[(dataframe[column] < lower_bound) | 
                            (dataframe[column] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(dataframe)) * 100
        
        results.append({
            'Feature': column,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound,
            'Outlier_Count': outlier_count,
            'Outlier_Percentage': f"{outlier_percentage:.2f}%"
        })
    
    return pd.DataFrame(results)

# Apply outlier detection
outliers_summary = detect_outliers_iqr(df)
print("\nOutlier Detection Summary (IQR Method):")
print(outliers_summary.to_string(index=False))

# Save outlier summary to CSV
output_path = os.path.join(OUTPUT_DIR, 'outliers_summary.csv')
outliers_summary.to_csv(output_path, index=False)
print(f"\n✓ Outlier summary saved as '{output_path}'")

# Visualize outliers using boxplots
print("\nCreating boxplots to visualize outliers...")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

for idx, column in enumerate(df.columns):
    axes[idx].boxplot(df[column], vert=True)
    axes[idx].set_title(f'Boxplot of {column}', fontsize=10, fontweight='bold')
    axes[idx].set_ylabel(column)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, 'boxplots_outliers.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Boxplots saved as '{output_path}'")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("EDA SUMMARY")
print("="*70)
print(f"Total samples: {len(df)}")
print(f"Total features (including target): {len(df.columns)}")
print(f"Total features (excluding target): {len(df.columns) - 1}")
print(f"\nTop 3 features most correlated with Price:")
for i, feature in enumerate(top_3_features, 1):
    print(f"  {i}. {feature}: {target_correlations[feature]:.4f}")

print(f"\nTotal outliers detected (cumulative across all features):")
total_outliers = outliers_summary['Outlier_Count'].sum()
print(f"  {total_outliers} outlier instances")

print("\n" + "="*70)
print("All visualizations have been saved to 'outputs' folder!")
print("="*70)
