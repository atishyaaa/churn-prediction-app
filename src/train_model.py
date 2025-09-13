# src/train_model.py

import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.dummy import DummyClassifier

# Import custom FeatureEngineer for joblib load
from featureengineer import FeatureEngineer

# -------------------------------
# Paths
# -------------------------------
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'randomdata.csv')
PREPROCESSOR_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'preprocessor.pkl')
MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODELS_PATH, exist_ok=True)

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(DATA_PATH)
y = df['Churn'].map({'Yes': 1, 'No': 0})
X = df.drop(columns=['Churn'])

# -------------------------------
# Load preprocessor and transform
# -------------------------------
print("Loading preprocessor and transforming data...")
pipeline = joblib.load(PREPROCESSOR_PATH)
X_processed = pipeline.transform(X)

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Dummy baseline
# -------------------------------
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
y_dummy_pred = dummy.predict(X_test)
print(f"Dummy classifier accuracy: {accuracy_score(y_test, y_dummy_pred):.4f}\n")

# -------------------------------
# Define models with minimal changes
# -------------------------------
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=5, max_features='sqrt',
        random_state=42, class_weight='balanced'
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=3, min_samples_leaf=5, random_state=42
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced', C=0.1
    )
}

# -------------------------------
# K-Fold CV and training
# -------------------------------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("Training models with 5-fold cross-validation...\n")
for name, model in models.items():
    # Convert sparse to dense if needed for LogisticRegression
    X_train_input = X_train.toarray() if name == "LogisticRegression" and hasattr(X_train, "toarray") else X_train

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_input, y_train, cv=kf, scoring='accuracy')
    print(f"{name} CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Fit full training set
    model.fit(X_train_input, y_train)

    X_test_input = X_test.toarray() if name == "LogisticRegression" and hasattr(X_test, "toarray") else X_test
    y_pred = model.predict(X_test_input)

    # Store metrics
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred)
    }

# -------------------------------
# Show results and select best model
# -------------------------------
print("\nModel Performance Summary:\n")
best_model = None
best_score = 0

for name, metrics in results.items():
    print(f"{name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print()
    if metrics['roc_auc'] > best_score:
        best_score = metrics['roc_auc']
        best_model = models[name]

# -------------------------------
# Save best model
# -------------------------------
model_path = os.path.join(MODELS_PATH, f"{best_model.__class__.__name__}_model.pkl")
joblib.dump(best_model, model_path)
print(f"Best Model: {best_model.__class__.__name__}")
print(f"Saved at: {model_path}")

# -------------------------------
# Feature Importance for tree-based models
# -------------------------------
if isinstance(best_model, (RandomForestClassifier, GradientBoostingClassifier)) and hasattr(best_model, "feature_importances_"):
    print("\nTop 10 Feature Importances:")
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    importances = best_model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    for idx in top_idx:
        print(f"{feature_names[idx]}: {importances[idx]:.4f}")
