# Customer Churn Prediction App

A machine learning powered **Streamlit web app** that predicts whether a customer is likely to churn.  
This project demonstrates **end-to-end ML skills** including feature engineering, model training, evaluation, and deployment.

---

## 🚀 Features

- Interactive **Streamlit UI** for real-time predictions
- Single-customer churn prediction with probability scores
- **Batch CSV upload** with downloadable results
- Visual insights (confusion matrix, churn distribution charts)
- Custom feature engineering pipeline
- Model comparison and selection
- Comprehensive performance metrics

---

## 📁 Project Structure

```
churn-prediction-app/
├── app/                              # Streamlit web application
│   └── app.py                        # Main Streamlit app
│
├── src/                              # Source code for training & evaluation
│   ├── featureengineer.py           # Custom feature engineering pipeline
    ├── preprocess.py                # Script to preprocess the model for training
│   ├── train_model.py               # Script to train model(s)
│   └── evaluate.py                  # Script to evaluate trained models
│
├── models/                          # Saved models & reports
│   ├── RandomForestClassifier_model.pkl
│   ├── GradientBoostingClassifier_model.pkl
│   ├── LogisticRegression_model.pkl
│   ├── preprocessor.pkl
│   ├── metrics_report.csv
│   └── predictions.csv
│
├── data/                            
│   └── randomdata.csv
    ├── RandomForestClassifier_model.pkl      # Example input file for batch predictions
│
├── notebooks/                       # Jupyter notebooks (optional)
│   └── eda.ipynb                    # EDA and model experimentation
│
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
└── .gitignore                      # Files to ignore in Git
```

---

## 📊 Model Performance

- **Best Algorithm:** Random Forest Classifier
- **Accuracy:** ~89%  
- **Precision:** ~87%
- **Recall:** ~85%
- **F1-score:** ~88%  

---

## 💻 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/churn-prediction-app.git
cd churn-prediction-app
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Models (Optional)
```bash
# Train all models and generate evaluation reports
python src/train_model.py

# Evaluate model performance
python src/evaluate.py
```

### 4. Run the Streamlit App
```bash
streamlit run app/app.py
```

### 5. Access the App
Open your browser and navigate to `http://localhost:8501`

---

## 📝 Usage Examples

### Single Customer Prediction
1. Navigate to the "Predict Churn" tab
2. Fill in customer details using the interactive form
3. Click "Predict Churn" to get results with probability scores

### Batch Predictions
1. Go to the "Batch Prediction" tab
2. Upload a CSV file with customer data
3. Download results with churn predictions and probabilities

### Sample Data Format
Use the included dataset to test batch predictions:
```bash
data/randomdata.csv
data/dummydata.csv
```

**Output format:**
- `churn_prediction`: 0 = No Churn, 1 = Churn
- `churn_probability`: Probability score (0-1)

---

## 🛠️ Technical Details

### Dependencies
- **Python 3.8+**
- **Streamlit** - Web app framework
- **Scikit-learn** - Machine learning models
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization
- **Joblib** - Model serialization

### Feature Engineering
- Automated preprocessing pipeline
- Categorical encoding (One-hot, Label encoding)
- Numerical scaling (StandardScaler)
- Missing value imputation
- Feature selection based on importance scores

### Models Implemented
- **Random Forest Classifier** (Best performer)
- **Gradient Boosting Classifier**
- **Logistic Regression**
- Support for easy addition of new models

---

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set model path
export MODEL_PATH="models/RandomForestClassifier_model.pkl"

# Optional: Set data path
export DATA_PATH="data/"
```

### Model Retraining
To retrain models with new data:
1. Place training data in `data/train.csv`
2. Run `python src/train_model.py`
3. New models will be saved in `models/` directory

---

## 📈 Future Enhancements

- [ ] Real-time model monitoring and drift detection
- [ ] A/B testing framework for model comparison
- [ ] Integration with cloud services (AWS, GCP, Azure)
- [ ] Advanced visualization dashboards
- [ ] API endpoint for programmatic access
- [ ] Docker containerization
- [ ] Automated model retraining pipeline

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🌟 Why This Project?

This project demonstrates:
- ✅ **End-to-end ML lifecycle** – from preprocessing to deployment
- ✅ **Reusable feature engineering pipelines** for production use
- ✅ **Interactive web deployment** with Streamlit
- ✅ **Clean, production-ready project structure**
- ✅ **Model evaluation and comparison** methodologies
- ✅ **Batch processing capabilities** for business use cases
- ✅ **Visual insights and interpretability** for stakeholders

---

## 👩‍💻 Author

**Atishya Pradhan** – Aspiring Data Scientist  
📧 [pradhanatishya@gmail.com](mailto:pradhanatishya@gmail.com)  
💼 [LinkedIn](www.linkedin.com/in/atishyapradhan) 

---

## 🙏 Acknowledgments

- Dataset inspired by insurance industry churn analysis
- Built with open-source tools and frameworks
- Special thanks to the Streamlit and Scikit-learn communities

---

**⭐ Star this repo if you found it helpful!**