
# Advanced Loan Approval Predictor

## Project Overview

**Advanced Loan Approval Predictor** is a supervised machine learning project that predicts whether a loan application should be approved based on applicant financial data.

The system is trained on a real applicant dataset (`loans.xlsx`) containing 1,000 records, using four features — income, credit score, requested loan amount, and years of employment — to predict loan approval status. A Random Forest classifier is trained on this data after preprocessing (train/test split and feature scaling), achieving **95% accuracy** on the held-out test set.

The trained model is served through a FastAPI backend (`main.py`) and exposed via a Streamlit web interface (`app.py`), allowing users to enter applicant details and receive an instant loan approval prediction.

**Tech stack:** Python, pandas, scikit-learn, FastAPI, Streamlit, joblib

---

## Dataset Overview

The model is trained on a real applicant dataset supplied as an Excel file, `loans.xlsx`. `train_model.py` loads it directly using pandas' `read_excel()`.

**Source File:** `loans.xlsx` — loaded with `df = pd.read_excel('loans.xlsx')`. The dataset contains 1,000 applicant records across 5 columns (4 input features + 1 target label).

---

## Model Training

A `RandomForestClassifier` (scikit-learn, default hyperparameters, `random_state=42`) is trained on the scaled training data. Random Forest was a reasonable choice here since it handles non-linear relationships between income, credit score, loan amount, and employment history without requiring manual feature engineering.

After training, the model and scaler are exported as `model.pkl` and `scaler.pkl` respectively using `joblib`, so the FastAPI backend (`main.py`) can load them without retraining.

---

## Evaluation — Accuracy

**Accuracy:** The proportion of test-set applicants (out of the 200 held out for testing, i.e. 20% of 1,000 rows) whose loan decision the model predicted correctly.

**Result: 0.95 (95%)**

An accuracy of 0.95 means that, on the held-out test set, the Random Forest model's approve/reject decision matched the dataset's labelled outcome for 190 out of 200 applicants.
