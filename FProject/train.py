import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             precision_score, recall_score, f1_score, auc)
import joblib
import warnings
import os
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Create models directory
os.makedirs('models', exist_ok=True)
warnings.filterwarnings('ignore')

#LOAD AND PREPARE DATA
print("Loading data...")
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "data_shared_Diabetes project.csv")
features = [
    'FBG',
    'HbA1c',
    'BMI',
    'Age',
    'ParentalDiabetesHistory'
]

target = 'DiabetesType'

X = df[features].copy()
y = df[target].copy()

print("Handling missing values...")
for col in features:
    if X[col].dtype in ['float64', 'int64']:
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].fillna(X[col].mode()[0])

valid_rows = y.notna()
X = X[valid_rows]
y = y[valid_rows]
y = y.astype(int)

print(f"\nDataset shape: {X.shape}")
print(f"Diabetes distribution:\n{y.value_counts()}")
print(f"Diabetes rate: {y.mean():.2%}")

#FEATURE ENGINEERING
print("\nCreating additional features...")

X['BMI_Age'] = X['BMI'] * X['Age'] / 100
X['Glucose_HbA1c'] = X['FBG'] * X['HbA1c'] / 100
X['Metabolic_Risk'] = (X['FBG'] / 100 + X['HbA1c'] / 10 + X['BMI'] / 30) / 3

for col in ['FBG', 'HbA1c']:
    X[f'Log_{col}'] = np.log1p(X[col])

#REMOVE OUTLIERS
print("\nRemoving outliers...")
def remove_outliers(df, columns, threshold=3):
    z_scores = np.abs((df[columns] - df[columns].mean()) / df[columns].std())
    return df[(z_scores < threshold).all(axis=1)]

original_len = len(X)
X = remove_outliers(X, ['BMI', 'Age', 'FBG'])
y = y[X.index]
print(f"Removed {original_len - len(X)} outliers")

feature_names = X.columns.tolist()

#SCALE FEATURES
print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

#TRAIN INDIVIDUAL MODELS
print("\n" + "=" * 60)
print("Training Individual base models")
print("=" * 60)

# Logistic Regression
print("\n1. Training Logistic Regression...")
lr_model = LogisticRegression(
    C=1.0, penalty='l2', class_weight='balanced',
    random_state=42, max_iter=1000
)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_proba = lr_model.predict_proba(X_test)[:, 1]
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_proba)
lr_cm = confusion_matrix(y_test, lr_pred)

print(f"   Accuracy: {lr_accuracy:.4f}")
print(f"   Precision: {lr_precision:.4f}")
print(f"   Recall: {lr_recall:.4f}")
print(f"   F1-Score: {lr_f1:.4f}")
print(f"   AUC-ROC: {lr_auc:.4f}")

# Decision Tree
print("\n2. Training Decision Tree...")
dt_model = DecisionTreeClassifier(
    max_depth=5, min_samples_split=10,
    min_samples_leaf=5, random_state=42
)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_proba = dt_model.predict_proba(X_test)[:, 1]
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)
dt_auc = roc_auc_score(y_test, dt_proba)
dt_cm = confusion_matrix(y_test, dt_pred)

print(f"   Accuracy: {dt_accuracy:.4f}")
print(f"   Precision: {dt_precision:.4f}")
print(f"   Recall: {dt_recall:.4f}")
print(f"   F1-Score: {dt_f1:.4f}")
print(f"   AUC-ROC: {dt_auc:.4f}")

# Random Forest
print("\n3. Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=10,
    min_samples_split=5, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_proba)
rf_cm = confusion_matrix(y_test, rf_pred)

print(f"   Accuracy: {rf_accuracy:.4f}")
print(f"   Precision: {rf_precision:.4f}")
print(f"   Recall: {rf_recall:.4f}")
print(f"   F1-Score: {rf_f1:.4f}")
print(f"   AUC-ROC: {rf_auc:.4f}")

#BUILD AND TRAIN STACKING ENSEMBLE
print("\n" + "=" * 60)
print("BUILDING STACKING ENSEMBLE")
print("=" * 60)

print("\nBase Models: Logistic Regression, Decision Tree, Random Forest")
print("Meta Model: Random Forest")

base_models = [
    ('logistic_regression', LogisticRegression(
        C=1.0, penalty='l2', class_weight='balanced',
        random_state=42, max_iter=1000
    )),
    ('decision_tree', DecisionTreeClassifier(
        max_depth=5, min_samples_split=10,
        min_samples_leaf=5, random_state=42
    )),
    ('random_forest', RandomForestClassifier(
        n_estimators=100, max_depth=10,
        min_samples_split=5, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    ))
]

meta_model = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    min_samples_split=5, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    stack_method='predict_proba',
    n_jobs=-1
)

print("\nTraining stacking ensemble model...")
stacking_model.fit(X_train, y_train)

stacking_pred = stacking_model.predict(X_test)
stacking_proba = stacking_model.predict_proba(X_test)[:, 1]
stacking_accuracy = accuracy_score(y_test, stacking_pred)
stacking_precision = precision_score(y_test, stacking_pred)
stacking_recall = recall_score(y_test, stacking_pred)
stacking_f1 = f1_score(y_test, stacking_pred)
stacking_auc = roc_auc_score(y_test, stacking_proba)
stacking_cm = confusion_matrix(y_test, stacking_pred)

print(f"\n   Accuracy: {stacking_accuracy:.4f}")
print(f"   Precision: {stacking_precision:.4f}")
print(f"   Recall: {stacking_recall:.4f}")
print(f"   F1-Score: {stacking_f1:.4f}")
print(f"   AUC-ROC: {stacking_auc:.4f}")

#MODEL COMPARISON SUMMARY
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)

comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Stacking Ensemble'],
    'Accuracy': [lr_accuracy, dt_accuracy, rf_accuracy, stacking_accuracy],
    'Precision': [lr_precision, dt_precision, rf_precision, stacking_precision],
    'Recall': [lr_recall, dt_recall, rf_recall, stacking_recall],
    'F1-Score': [lr_f1, dt_f1, rf_f1, stacking_f1],
    'AUC-ROC': [lr_auc, dt_auc, rf_auc, stacking_auc]
})

print(comparison_df.to_string(index=False))

#SAVE MODELS
print("\n" + "=" * 60)
print("SAVING MODELS")
print("=" * 60)

joblib.dump(stacking_model, 'models/stacking_model.pkl')
joblib.dump(lr_model, 'models/lr_model.pkl')
joblib.dump(dt_model, 'models/dt_model.pkl')
joblib.dump(rf_model, 'models/rf_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(feature_names, 'models/feature_names.pkl')

feature_info = {
    'names': feature_names,
    'n_features': len(feature_names),
    'target': target
}
joblib.dump(feature_info, 'models/feature_info.pkl')

print("Stacking model saved to 'models/stacking_model.pkl'")
print("Logistic Regression saved to 'models/lr_model.pkl'")
print("Decision Tree saved to 'models/dt_model.pkl'")
print("Random Forest saved to 'models/rf_model.pkl'")
print("Scaler saved to 'models/scaler.pkl'")
print("Feature names saved to 'models/feature_names.pkl'")

#GENERATE PLOTS AND GRAPHS
print("\n" + "=" * 60)
print("Generating Evaluation Graphs")
print("=" * 60)

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Colour formatting
COLORS = {
    'primary': '#2C3E50',
    'secondary': '#E74C3C',
    'tertiary': '#27AE60',
    'quaternary': '#F39C12',
    'purple': '#8E44AD',
    'teal': '#1ABC9C',
    'gray': '#7F8C8D',
    'gold': '#D4AC0D'
}
#CONFUSION MATRIX
print("\nGenerating confusion matrix...")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(stacking_cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'],
            linewidths=1, linecolor='white',
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 14, 'weight': 'bold'})
ax.set_title('Confusion Matrix - Stacking Ensemble', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Confusion matrix saved")

#ROC CURVE
print("Generating ROC curve...")
fig, ax = plt.subplots(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, stacking_proba)
roc_auc = auc(fpr, tpr)

ax.plot(fpr, tpr, color=COLORS['secondary'], lw=2.5, label=f'Stacking Ensemble (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color=COLORS['gray'], lw=1.5, linestyle='--', alpha=0.7)
ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS['secondary'])
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='gray', framealpha=0.9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('models/roc_curve.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("ROC curve saved")

#PRECISION-RECALL CURVE
print("Generating Precision-Recall curve...")
fig, ax = plt.subplots(figsize=(8, 6))
precision_vals, recall_vals, _ = precision_recall_curve(y_test, stacking_proba)
pr_auc = auc(recall_vals, precision_vals)

ax.plot(recall_vals, precision_vals, color=COLORS['tertiary'], lw=2.5, label=f'PR Curve (AUC = {pr_auc:.4f})')
ax.fill_between(recall_vals, precision_vals, alpha=0.15, color=COLORS['tertiary'])
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc="lower left", frameon=True, facecolor='white', edgecolor='gray', framealpha=0.9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('models/pr_curve.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("PR curve saved")

#METRICS COMPARISON
print("Generating metrics comparison...")
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
values = [stacking_accuracy, stacking_precision, stacking_recall, stacking_f1, stacking_auc]
bar_colors = ['#2C3E50', '#E74C3C', '#27AE60', '#F39C12', '#8E44AD']

bars = ax.bar(metrics, values, color=bar_colors, edgecolor='#2C3E50', linewidth=1.2, width=0.7)
ax.set_ylim([0, 1.0])
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Stacking Ensemble - Performance Metrics', fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.15, axis='y')
ax.set_axisbelow(True)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='regular')
plt.tight_layout()
plt.savefig('models/metrics_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Metrics comparison saved")

#MODEL ACCURACY COMPARISON
print("Generating model accuracy comparison (horizontal bar chart)...")
fig, ax = plt.subplots(figsize=(10, 6))
model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Stacking Ensemble']
accuracies = [lr_accuracy, dt_accuracy, rf_accuracy, stacking_accuracy]
bar_colors = ['#2C3E50', '#E74C3C', '#F39C12', '#27AE60']

# Horizontal bar chart
bars = ax.barh(model_names, accuracies, color=bar_colors, edgecolor='#2C3E50', linewidth=1.5, height=0.6)
ax.set_xlim([0, 1.0])
ax.set_xlabel('Accuracy Score', fontsize=12)
ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.15, axis='x')
ax.set_axisbelow(True)

# Add value labels
for bar, val in zip(bars, accuracies):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

# Best accuracy line
best_acc = max(accuracies)
ax.axvline(x=best_acc, color='#27AE60', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(best_acc + 0.02, 0.5, f'Best: {best_acc:.3f}', fontsize=9, fontweight='bold', color='#27AE60')

plt.tight_layout()
plt.savefig('models/model_accuracy_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Accuracy comparison saved to 'models/model_accuracy_comparison.png'")

#FEATURE IMPORTANCE
print("Generating feature importance pie chart...")

importances = stacking_model.final_estimator_.feature_importances_
indices = np.argsort(importances)[::-1]

# Get top 5 features (clean and readable)
n_features = min(5, len(feature_names))
top_indices = indices[:n_features]
top_names = [feature_names[i] for i in top_indices]
top_importances = importances[top_indices]

# If there are more features, group the rest as "Others"
if len(feature_names) > 5:
    other_importance = sum(importances[indices[n_features:]])
    if other_importance > 0.01:  # Only add if significant
        top_names.append('Others')
        top_importances = np.append(top_importances, other_importance)

fig, ax = plt.subplots(figsize=(7, 7))

# Simple clean colors
colors = ['#2C3E50', '#E74C3C', '#27AE60', '#F39C12', '#8E44AD', '#7F8C8D']

ax.pie(top_importances, labels=top_names, autopct='%1.1f%%',
       startangle=90, colors=colors[:len(top_names)])

ax.set_title('Feature Importance Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('models/feature_importance.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Feature importance pie chart saved to 'models/feature_importance.png'")

#CROSS-VALIDATION SCORES
print("Generating cross-validation scores...")
cv_scores = cross_val_score(stacking_model, X_scaled, y, cv=5, scoring='accuracy')

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(range(1, 6), cv_scores, color=COLORS['primary'], edgecolor='#2C3E50', linewidth=1.2, width=0.6)
ax.axhline(y=cv_scores.mean(), color=COLORS['secondary'], linestyle='--', linewidth=1.5,
            label=f'Mean CV Score: {cv_scores.mean():.4f}')
ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('5-Fold Cross-Validation', fontsize=14, fontweight='bold', pad=20)
ax.set_ylim([0, 1.0])
ax.set_xticks(range(1, 6))
ax.grid(True, alpha=0.15, axis='y')
ax.set_axisbelow(True)

for bar, val in zip(bars, cv_scores):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)

ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='gray')
plt.tight_layout()
plt.savefig('models/cv_scores.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("CV scores saved")

#CLASSIFICATION REPORT HEATMAP
print("Generating classification report heatmap...")

# Generate the report with only class names
report = classification_report(y_test, stacking_pred, target_names=['No Diabetes', 'Diabetes'], output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Keep ALL rows (classes + accuracy + averages)
# Remove 'support' column (keep only precision, recall, f1-score)
report_df = report_df.iloc[:, :3]  # Keep only first 3 columns

# Rename index for better display
report_df.index = ['No Diabetes', 'Diabetes', 'Accuracy', 'Macro Avg', 'Weighted Avg']

fig, ax = plt.subplots(figsize=(10, 5.5))
sns.heatmap(report_df, annot=True, fmt='.3f', cmap='YlOrBr',
            xticklabels=['Precision', 'Recall', 'F1-Score'],
            yticklabels=report_df.index.tolist(),
            annot_kws={'size': 11, 'weight': 'bold'},
            cbar_kws={'label': 'Score', 'shrink': 0.8},
            linewidths=1.5, linecolor='white')
ax.set_title('Classification Report', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('models/classification_report_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Classification report saved")

#PREDICTION DISTRIBUTION
print("Generating prediction distribution...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

actual_counts = pd.Series(y_test).value_counts()
bars1 = ax1.bar(['No Diabetes', 'Diabetes'], actual_counts.values,
            color=['#2C3E50', '#E74C3C'], edgecolor='white', linewidth=1.5)
ax1.set_title('Actual Distribution', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count', fontsize=11)
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(int(bar.get_height())), ha='center', va='bottom', fontsize=10, fontweight='bold')

pred_counts = pd.Series(stacking_pred).value_counts()
bars2 = ax2.bar(['No Diabetes', 'Diabetes'], pred_counts.values,
            color=['#2C3E50', '#E74C3C'], edgecolor='white', linewidth=1.5)
ax2.set_title('Predicted Distribution', fontsize=12, fontweight='bold')
ax2.set_ylabel('Count', fontsize=11)
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(int(bar.get_height())), ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.grid(True, alpha=0.15, axis='y')
ax2.grid(True, alpha=0.15, axis='y')
ax1.set_axisbelow(True)
ax2.set_axisbelow(True)
plt.suptitle('Prediction Distribution - Stacking Ensemble', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('models/prediction_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Prediction distribution saved")

#SHAP ANALYSIS
print("\n" + "=" * 60)
print("SHAP ANALYSIS")
print("=" * 60)

final_model = stacking_model.final_estimator_
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)

print("\nGenerating SHAP summary plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values[:, :, 1], X_test, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig('models/shap_summary_plot.png', bbox_inches='tight', dpi=150, facecolor='white')
plt.close()
print("SHAP summary plot saved")

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values[:, :, 1], X_test, feature_names=feature_names, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('models/shap_bar_plot.png', bbox_inches='tight', dpi=150, facecolor='white')
plt.close()
print("SHAP bar plot saved")

shap_data = {
    'shap_values': shap_values,
    'feature_names': feature_names,
    'X_test_sample': X_test[:5],
    'expected_value': explainer.expected_value[1]
}
joblib.dump(shap_data, 'models/shap_data.pkl')
print("SHAP data saved")

#FINAL SUMMARY
print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)

print("\nModel Architecture:")
print("   Base Models (Level 0):")
print("   - Logistic Regression")
print("   - Decision Tree")
print("   - Random Forest")
print("   Meta Model (Level 1):")
print("   - Random Forest (200 estimators)")

print("\nPerformance Summary:")
print("-" * 60)
print(f"   {'Model':<20} {'Accuracy':<12} {'AUC-ROC':<12}")
print("-" * 60)
print(f"   {'Logistic Regression':<20} {lr_accuracy:.4f}      {lr_auc:.4f}")
print(f"   {'Decision Tree':<20} {dt_accuracy:.4f}      {dt_auc:.4f}")
print(f"   {'Random Forest':<20} {rf_accuracy:.4f}      {rf_auc:.4f}")
print(f"   {'Stacking Ensemble':<20} {stacking_accuracy:.4f}      {stacking_auc:.4f} (Best)")
print("-" * 60)

print("\nGraphs Saved:")
print("   - confusion_matrix.png")
print("   - roc_curve.png")
print("   - pr_curve.png")
print("   - metrics_comparison.png")
print("   - model_accuracy_comparison.png")
print("   - feature_importance.png")
print("   - cv_scores.png")
print("   - classification_report_heatmap.png")
print("   - prediction_distribution.png")
print("   - shap_summary_plot.png")
print("   - shap_bar_plot.png")
