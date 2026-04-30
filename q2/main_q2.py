"""
Question 2: Main Orchestrator for Extreme Class Imbalance Pipeline
Coordinates all phases: EDA, Modeling, Calibration, and Threshold Tuning
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# Set output directory - Q2 outputs go to outputs/q2/
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'q2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import functions from our modules
from credit_card_fraud_analysis import fetch_and_clean_data, calculate_imbalance_ratio, plot_class_distribution
from imbalanced_classifier_evaluation import ImbalancedClassifierEvaluator
from calibration_analysis import CalibrationAnalyzer
from threshold_tuning import ThresholdTuner

print("\n" + "=" * 80)
print("EXTREME CLASS IMBALANCE PIPELINE - QUESTION 2")
print("=" * 80)
print(f"\nOutput directory: {OUTPUT_DIR}")

# ============================================================================
# PHASE 1: DATA FETCHING AND EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "█" * 80)
print("PHASE 1: DATA FETCHING AND EXPLORATORY DATA ANALYSIS")
print("█" * 80)

print("\n[1.1] Fetching Credit Card Fraud dataset...")
df = fetch_and_clean_data()

print("\n[1.2] Calculating Imbalance Ratio...")
imbalance_ratio, class_counts = calculate_imbalance_ratio(df)

print("\n[1.3] Plotting class distribution...")
plot_class_distribution(class_counts)
print("  - class_imbalance.png")

# Prepare X and y for modeling
print("\n[1.4] Preparing features and target...")
X = df.drop(df.columns[-1], axis=1).values
y = df.iloc[:, -1].values

# Ensure proper data types
X = np.asarray(X, dtype=float)
y = np.asarray(y, dtype=int)

# Remove any rows with NaN values in X
mask = ~np.isnan(X).any(axis=1)
X = X[mask]
y = y[mask]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Class distribution: {np.bincount(y)}")

# ============================================================================
# TRAIN/TEST SPLIT (STRATIFIED TO MAINTAIN IMBALANCE)
# ============================================================================
print("\n[1.5] Performing stratified Train/Test split (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Training class distribution: {np.bincount(y_train)}")
print(f"Test class distribution: {np.bincount(y_test)}")

# Verify imbalance is maintained
train_ir = np.sum(y_train == 0) / np.sum(y_train == 1)
test_ir = np.sum(y_test == 0) / np.sum(y_test == 1)
print(f"Training Imbalance Ratio: {train_ir:.4f}")
print(f"Test Imbalance Ratio: {test_ir:.4f}")

# ============================================================================
# PHASE 2 & 3: CLASSIFIER EVALUATION WITH RESAMPLING STRATEGIES
# ============================================================================
print("\n" + "█" * 80)
print("PHASE 2 & 3: CLASSIFIER EVALUATION WITH RESAMPLING STRATEGIES")
print("█" * 80)

print("\n[2.1] Initializing classifier evaluator...")
evaluator = ImbalancedClassifierEvaluator(X_train, y_train, n_splits=5, random_state=42)

print("\n[2.2] Evaluating all classifiers...")
evaluator.evaluate_all()

# Get results dataframe
results_df = evaluator.get_results_dataframe()

print("\n[2.3] Results Summary:")
print(results_df.to_string(index=False))

# Save results
classifier_results_path = os.path.join(OUTPUT_DIR, 'classifier_evaluation_results.csv')
results_df.to_csv(classifier_results_path, index=False)
print(f"\nResults saved to '{classifier_results_path}'")

# ============================================================================
# PHASE 4: PROBABILITY CALIBRATION
# ============================================================================
print("\n" + "█" * 80)
print("PHASE 4: PROBABILITY CALIBRATION")
print("█" * 80)

print("\n[4.1] Training best models on full training set for calibration...")

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Calculate scale_pos_weight for XGBoost
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

# Train RandomForest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("RandomForest trained")

# Train XGBoost
xgb_model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    tree_method='hist',
    verbosity=0
)
xgb_model.fit(X_train, y_train)
print("XGBoost trained")

print("\n[4.2] Performing calibration analysis...")
models_dict = {
    'XGBoost': xgb_model,
    'RandomForest': rf_model
}

calibration_analyzer = CalibrationAnalyzer(models_dict, X_train, X_test, y_train, y_test)
calibration_results = calibration_analyzer.analyze_calibration()

print("\n[4.3] Generating calibration visualizations...")
calibration_analyzer.plot_calibration_curves()
calibration_analyzer.plot_combined_calibration_curve()

print("\n[4.4] Calibration Results Summary:")
print(calibration_results.to_string(index=False))
calibration_results_path = os.path.join(OUTPUT_DIR, 'calibration_results.csv')
calibration_results.to_csv(calibration_results_path, index=False)
print(f"\nCalibration results saved to '{calibration_results_path}'")

# Determine best calibrated model
print("\n[4.5] Selecting best performing calibrated model...")
xgb_sigmoid_brier = calibration_results[
    (calibration_results['Model'] == 'XGBoost') &
    (calibration_results['Calibration'] == 'Sigmoid (Platt)')
]['Brier Score'].values[0]

xgb_isotonic_brier = calibration_results[
    (calibration_results['Model'] == 'XGBoost') &
    (calibration_results['Calibration'] == 'Isotonic')
]['Brier Score'].values

rf_sigmoid_brier = calibration_results[
    (calibration_results['Model'] == 'RandomForest') &
    (calibration_results['Calibration'] == 'Sigmoid (Platt)')
]['Brier Score'].values[0]

rf_isotonic_brier = calibration_results[
    (calibration_results['Model'] == 'RandomForest') &
    (calibration_results['Calibration'] == 'Isotonic')
]['Brier Score'].values

# Find the model with lowest Brier score
brier_scores = {
    'XGBoost_Sigmoid': xgb_sigmoid_brier,
    'XGBoost_Isotonic': xgb_isotonic_brier[0] if len(xgb_isotonic_brier) > 0 else float('inf'),
    'RandomForest_Sigmoid': rf_sigmoid_brier,
    'RandomForest_Isotonic': rf_isotonic_brier[0] if len(rf_isotonic_brier) > 0 else float('inf')
}

best_model_name = min(brier_scores, key=brier_scores.get)
best_brier = brier_scores[best_model_name]

print(f"\nBrier Scores for all calibrated models:")
for model, score in sorted(brier_scores.items(), key=lambda x: x[1]):
    print(f"  {model}: {score:.6f}")

print(f"\nBest performing model: {best_model_name} (Brier Score: {best_brier:.6f})")

# Get the best calibrated model from the analyzer
best_calibrated_model = calibration_analyzer.calibrated_models[best_model_name]
best_predictions = calibration_analyzer.predictions[best_model_name]

# ============================================================================
# PHASE 5: THRESHOLD TUNING AND FINAL PREDICTIONS
# ============================================================================
print("\n" + "█" * 80)
print("PHASE 5: THRESHOLD TUNING AND FINAL PREDICTIONS")
print("█" * 80)

print(f"\n[5.1] Using best calibrated model: {best_model_name}")

print("\n[5.2] Analyzing precision-recall trade-offs...")
threshold_tuner = ThresholdTuner(y_test, best_predictions, model_name=best_model_name)

print("\n[5.3] Calculating optimal threshold...")
threshold_tuner.calculate_pr_curve()
threshold_tuner.find_optimal_threshold()

print("\n[5.4] Generating Precision-Recall curve...")
threshold_tuner.plot_pr_curve()

print("\n[5.5] Applying optimal threshold...")
y_pred_optimal = threshold_tuner.apply_optimal_threshold()

print("\n[5.6] Generating and plotting confusion matrix...")
cm, metrics = threshold_tuner.plot_confusion_matrix(y_pred_optimal=y_pred_optimal)

print("\n[5.7] Generating threshold analysis plot...")
threshold_tuner.plot_threshold_analysis()

print("\n[5.8] Final Summary Report:")
threshold_tuner.generate_summary_report()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PIPELINE EXECUTION COMPLETE")
print("=" * 80)

print("\nGenerated Output Files:")
print("  Phase 1 - EDA:")
print("    ✓ outputs/q2/class_imbalance.png - Class distribution with log scale")
print("  Phase 2-3 - Modeling:")
print("    ✓ outputs/q2/classifier_evaluation_results.csv - Classifier comparison")
print("  Phase 4 - Calibration:")
print("    ✓ outputs/q2/calibration_curves.png - Individual calibration curves")
print("    ✓ outputs/q2/calibration_curves_combined.png - Combined calibration comparison")
print("    ✓ outputs/q2/calibration_results.csv - Calibration metrics")
print("  Phase 5 - Threshold Tuning:")
print("    ✓ outputs/q2/pr_curve.png - Precision-Recall curve")
print("    ✓ outputs/q2/confusion_matrix_optimal.png - Confusion matrix at optimal threshold")
print("    ✓ outputs/q2/threshold_analysis.png - Threshold trade-off analysis")

print("\nKey Results:")
print(f"  Original Imbalance Ratio: {imbalance_ratio:.4f}")
print(f"  Training Set IR: {train_ir:.4f}")
print(f"  Test Set IR: {test_ir:.4f}")
print(f"  Best Calibrated Model: {best_model_name}")
print(f"  Best Model Brier Score: {best_brier:.6f}")
print(f"  Optimal Decision Threshold: {threshold_tuner.optimal_threshold:.6f}")
print(f"  F1-Score at Optimal Threshold: {threshold_tuner.optimal_f1:.6f}")
print(f"  Final Confusion Matrix:")
print(f"    True Negatives: {cm[0, 0]}")
print(f"    False Positives: {cm[0, 1]}")
print(f"    False Negatives: {cm[1, 0]}")
print(f"    True Positives: {cm[1, 1]}")
print(f"  Final Metrics:")
print(f"    Precision: {metrics['precision']:.6f}")
print(f"    Recall: {metrics['recall']:.6f}")
print(f"    Specificity: {metrics['specificity']:.6f}")
print(f"    F1-Score: {metrics['f1_score']:.6f}")

print("\n" + "=" * 80)
