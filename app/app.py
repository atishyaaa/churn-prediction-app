import sys
import os

# -------------------------------
# Fix: ensure src/ is in Python path
# -------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    from featureengineer import FeatureEngineer  # required for joblib unpickling
except ImportError:
    FeatureEngineer = None  # safety fallback

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# Paths
# -------------------------------
MODEL_PATH = os.path.join("models", "RandomForestClassifier_model.pkl")
PREPROCESSOR_PATH = os.path.join("models", "preprocessor.pkl")
METRICS_PATH = os.path.join("models", "metrics_report.csv")

# -------------------------------
# Load artifacts with retry safety
# -------------------------------
@st.cache_resource
def load_artifacts():
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.warning(f"⚠️ Reloading artifacts with import fix. Error: {e}")
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
        from featureengineer import FeatureEngineer
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        model = joblib.load(MODEL_PATH)
    return preprocessor, model

preprocessor, model = load_artifacts()

# -------------------------------
# Sidebar navigation
# -------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Model Performance", "🧾 Predict Churn", "📥 Batch CSV Upload"]
)

# -------------------------------
# Home Page
# -------------------------------
if page == "🏠 Home":
    st.title("Customer Churn Prediction App")
    st.write("""
Welcome to the Customer Churn Prediction App!   
This tool helps you understand and predict customer churn using machine learning.  

Use the sidebar to explore:  
- **Model Performance** → Check accuracy, F1 score, and other evaluation metrics  
- **Predict Churn** → Enter details for one customer to see if they’re at risk  
- **Batch CSV Upload** → Upload your own dataset and generate churn predictions at scale  
""")

# -------------------------------
# Model Performance
# -------------------------------
elif page == "📊 Model Performance":
    st.title("📊 Model Performance Report")

    if os.path.exists(METRICS_PATH):
        metrics_df = pd.read_csv(METRICS_PATH)
        st.dataframe(metrics_df)

        # Optional confusion matrix image
        if os.path.exists("models/confusion_matrix.png"):
            st.image("models/confusion_matrix.png", caption="Confusion Matrix")
    else:
        st.warning("⚠️ Metrics report not found. Run `evaluate.py` first.")

# -------------------------------
# Predict Single Customer
# -------------------------------
elif page == "🧾 Predict Churn":
    st.title("🧾 Predict Customer Churn")
    st.write("Fill in customer details:")

    # Example inputs (match your dataset columns)
    claim_amount = st.number_input("Claim Amount", min_value=0, step=100)
    category_premium = st.number_input("Category Premium", min_value=0, step=10)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1)
    claim_reason = st.selectbox("Claim Reason", ["Accident", "Phone", "Travel", "Other"])
    claim_request = st.selectbox("Claim Request Output", ["Approved", "Pending", "Rejected"])
    confidentiality = st.selectbox("Data confidentiality", ["Very low", "Low", "Medium", "High"])
    company_name = st.selectbox("CompanyName", ["A", "B", "C"])

    if st.button("Predict"):
        input_df = pd.DataFrame([{
            "Claim Amount": claim_amount,
            "Category Premium": category_premium,
            "BMI": bmi,
            "Claim Reason": claim_reason,
            "Claim Request output": claim_request,
            "Data confidentiality": confidentiality,
            "CompanyName": company_name
        }])

        X_processed = preprocessor.transform(input_df)
        pred = model.predict(X_processed)[0]
        prob = model.predict_proba(X_processed)[0][1]

        st.subheader("✅ Prediction Result")
        st.write("⚠️ Customer **WILL Churn**" if pred == 1 else "❌ Customer **will NOT Churn**")
        st.progress(float(prob))
        st.write(f"Churn Probability: **{prob:.2f}**")

# -------------------------------
# Batch CSV Upload
# -------------------------------
elif page == "📥 Batch CSV Upload":
    st.title("📥 Batch Predictions from CSV")
    st.write("Upload a CSV file with the same columns as training data.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:", df.head())

        # Preprocess & predict
        X_processed = preprocessor.transform(df)
        preds = model.predict(X_processed)
        probs = model.predict_proba(X_processed)[:, 1]

        df["Predicted_Churn"] = preds
        df["Predicted_Probability"] = probs

        st.subheader("✅ Predictions")
        st.dataframe(df.head(20))  # show first 20 results

        # Pie chart
        st.subheader("📊 Churn Distribution")
        churn_counts = df["Predicted_Churn"].value_counts()
        st.write(churn_counts)

        fig, ax = plt.subplots()
        ax.pie(
            churn_counts,
            labels=["No Churn", "Churn"],
            autopct="%1.1f%%",
            colors=["#4CAF50", "#FF5722"],
            startangle=90,
            wedgeprops={"edgecolor": "white"}
        )
        ax.set_title("Churn vs No Churn Distribution")
        st.pyplot(fig)

        # Download option
        csv_download = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Predictions CSV",
            data=csv_download,
            file_name="batch_predictions.csv",
            mime="text/csv"
        )
