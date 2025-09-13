# src/preprocess.py

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from featureengineer import FeatureEngineer  

# -------------------------------
# 1. Paths
# -------------------------------
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'randomdata.csv')
MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODELS_PATH, exist_ok=True)

# -------------------------------
# 2. Load Dataset
# -------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Target encoding
y = df['Churn'].map({'Yes': 1, 'No': 0})
X = df.drop(columns=['Churn'])

# -------------------------------
# 3. Define Column Types
# -------------------------------
categorical_cols = ['Claim Reason', 'Data confidentiality', 'Claim Request output']
numeric_cols = ['Claim Amount', 'Category Premium', 'Premium/Amount Ratio', 'BMI', 
                'Premium_to_BMI', 'CompanyName_FE']

# -------------------------------
# 4. Preprocessing Pipeline
# -------------------------------
cat_transformer = OneHotEncoder(drop='first', sparse_output=True, handle_unknown='ignore')
num_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numeric_cols),
        ('cat', cat_transformer, categorical_cols)
    ],
    sparse_threshold=0.3
)

pipeline = Pipeline([
    ('feature_engineer', FeatureEngineer()),
    ('preprocessor', preprocessor)
])

# -------------------------------
# 5. Fit + Transform + Save
# -------------------------------
print("Fitting and transforming data (this may take a few seconds)...")
X_processed = pipeline.fit_transform(X)

joblib.dump(pipeline, os.path.join(MODELS_PATH, 'preprocessor.pkl'))

# -------------------------------
# 6. Summary
# -------------------------------
print("Preprocessing complete!")
print(f"Preprocessor saved at: {os.path.join(MODELS_PATH, 'preprocessor.pkl')}")
print("Processed feature shape:", X_processed.shape)
print("Memory-efficient sparse encoding applied for large categorical features.")
