# src/evaluate.py

import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt

# -------------------------------
# Paths
# -------------------------------
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'randomdata.csv')
PREPROCESSOR_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'preprocessor.pkl')
BEST_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'RandomForestClassifier_model.pkl')

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(DATA_PATH)
y = df['Churn'].map({'Yes': 1, 'No': 0})
X = df.drop(columns=['Churn'])

# -------------------------------
# Split data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Load preprocessor and model
# -------------------------------
print("Loading preprocessor and model...")
pipeline = joblib.load(PREPROCESSOR_PATH)
model = joblib.load(BEST_MODEL_PATH)

# -------------------------------
# Preprocess test set
# -------------------------------
X_test_processed = pipeline.transform(X_test)

# -------------------------------
# Make predictions
# -------------------------------
y_pred = model.predict(X_test_processed)
y_prob = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, "predict_proba") else None

# -------------------------------
# Metrics
# -------------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"

print("\nEvaluation Metrics on Test Set:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc if roc != 'N/A' else 'N/A'}")

# -------------------------------
# Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# -------------------------------
# ROC Curve
# -------------------------------
if y_prob is not None:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

# -------------------------------
# Precision-Recall Curve
# -------------------------------
if y_prob is not None:
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    plt.figure()
    plt.plot(recall_vals, precision_vals, marker='.')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()

# -------------------------------
# Feature Importances (for tree-based)
# -------------------------------
if hasattr(model, "feature_importances_"):
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    plt.figure(figsize=(8, 6))
    plt.barh(range(10), importances[top_idx][::-1], align='center')
    plt.yticks(range(10), feature_names[top_idx][::-1])
    plt.xlabel("Importance")
    plt.title("Top 10 Feature Importances")
    plt.show()

# -------------------------------
# Save predictions and metrics
# -------------------------------
results_df = X_test.copy()
results_df['True_Churn'] = y_test
results_df['Predicted_Churn'] = y_pred
if y_prob is not None:
    results_df['Predicted_Probability'] = y_prob

results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'predictions.csv')
results_df.to_csv(results_path, index=False)

metrics_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'metrics_report.csv')
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
    "Value": [acc, prec, rec, f1, roc if roc != "N/A" else np.nan]
})
metrics_df.to_csv(metrics_path, index=False)

print(f"\nPredictions saved at: {results_path}")
print(f"Metrics report saved at: {metrics_path}")
