# create_dummy_data.py

import pandas as pd
import numpy as np
import os

# -------------------------------
# Parameters
# -------------------------------
N_ROWS = 5000
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'dummydata.csv')
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

# -------------------------------
# Fake categorical options
# -------------------------------
companies = [f'Company_{i}' for i in range(1, 16)]
claim_reasons = ['Travel', 'Phone', 'Other']
data_confidentiality = ['Low', 'Medium', 'Very low']
claim_request_output = ['Pending', 'Completed']

# -------------------------------
# Generate dataframe
# -------------------------------
np.random.seed(42)

df = pd.DataFrame({
    'Unnamed: 0': range(N_ROWS),
    'Customer Name': [f'Customer_{i}' for i in range(N_ROWS)],
    'Customer_Address': [f'Address_{i}' for i in range(N_ROWS)],
    'Claim Amount': np.random.randint(1000, 10000, size=N_ROWS),
    'Category Premium': np.random.randint(500, 5000, size=N_ROWS),
    'Premium/Amount Ratio': np.round(np.random.uniform(0, 2, size=N_ROWS), 2),
    'BMI': np.round(np.random.uniform(18, 40, size=N_ROWS), 1),
    'Company Name': np.random.choice(companies, size=N_ROWS),
    'Claim Reason': np.random.choice(claim_reasons, size=N_ROWS),
    'Data confidentiality': np.random.choice(data_confidentiality, size=N_ROWS),
    'Claim Request output': np.random.choice(claim_request_output, size=N_ROWS),
    'Churn': np.random.choice(['Yes', 'No'], size=N_ROWS, p=[0.5, 0.5])
})

# -------------------------------
# Save to CSV
# -------------------------------
df.to_csv(DATA_PATH, index=False)
print(f"Dummy dataset generated at: {DATA_PATH}")
print(f"Shape: {df.shape}")
