"""
Question 2: Credit Card Fraud Dataset Analysis
Handles data fetching, cleaning, imbalance ratio calculation, and visualization
"""

import os
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import warnings

warnings.filterwarnings('ignore')

# Outputs are organized in outputs/q2/ subfolder
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'q2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Disable interactive plot display to keep pipeline running
SHOW_PLOTS = False

def fetch_and_clean_data():
    """
    Fetch the Credit Card Fraud dataset and clean it
    """
    print("Fetching Credit Card Fraud dataset...")
    # Fetch the dataset
    data = fetch_openml(data_id=1597, as_frame=True, parser='auto')
    df = data.frame
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"\nMissing values before cleaning:")
    print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values")
    
    # Clean the data - handle missing values
    if df.isnull().sum().sum() > 0:
        print("\nCleaning data: Removing rows with missing values...")
        df = df.dropna()
        print(f"Dataset shape after cleaning: {df.shape}")
    else:
        print("\nNo missing values found. Data is clean.")
    
    # Convert target variable to integers (handle both string and numeric types)
    target_col = df.columns[-1]
    if df[target_col].dtype == 'object' or df[target_col].dtype == 'str':
        print(f"\nConverting target variable '{target_col}' to integers...")
        df[target_col] = df[target_col].astype(int)
    
    # Ensure all numeric columns are properly typed
    print("\nConverting numeric columns to float...")
    for col in df.columns[:-1]:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    
    return df

def calculate_imbalance_ratio(df):
    """
    Calculate the Imbalance Ratio (IR)
    IR = number of majority class / number of minority class
    """
    # The last column is typically the target variable
    target_col = df.columns[-1]
    class_counts = df[target_col].value_counts()
    
    print(f"\nClass distribution:")
    print(class_counts)
    
    # Calculate IR
    majority_count = class_counts.max()
    minority_count = class_counts.min()
    imbalance_ratio = majority_count / minority_count
    
    print(f"\nImbalance Ratio (IR): {imbalance_ratio:.4f}")
    print(f"Majority class count: {majority_count}")
    print(f"Minority class count: {minority_count}")
    
    return imbalance_ratio, class_counts

def plot_class_distribution(class_counts):
    """
    Plot bar chart showing class distribution with logarithmic y-axis
    """
    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    classes = [str(c) for c in class_counts.index]
    counts = class_counts.values
    
    plt.bar(classes, counts, color=['#2E86AB', '#A23B72'], edgecolor='black', linewidth=1.5)
    
    # Set logarithmic scale for y-axis to make minority class visible
    plt.yscale('log')
    
    # Add labels and title
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Count (log scale)', fontsize=12, fontweight='bold')
    plt.title('Credit Card Fraud Dataset - Class Imbalance Distribution', 
              fontsize=14, fontweight='bold')
    
    # Add value labels on top of bars
    for i, (cls, count) in enumerate(zip(classes, counts)):
        plt.text(i, count, f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, 'class_imbalance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{output_path}'")
    
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

def main():
    """
    Main function to orchestrate the analysis
    """
    print("="*60)
    print("Credit Card Fraud Dataset Analysis")
    print("="*60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Step 1: Fetch and clean data
    df = fetch_and_clean_data()
    
    # Step 2: Calculate imbalance ratio
    imbalance_ratio, class_counts = calculate_imbalance_ratio(df)
    
    # Step 3: Plot class distribution
    plot_class_distribution(class_counts)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)

if __name__ == "__main__":
    print("Running as standalone script...")
    main()
