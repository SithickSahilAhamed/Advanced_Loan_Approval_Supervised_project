from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Updated to match the 4 variables exactly
class ApplicantInfo(BaseModel):
    income: float
    credit_score: float
    loan_amount: float
    employment_years: float

@app.post("/predict")
def predict_approval(applicant: ApplicantInfo):
    # Array must match the exact order trained in train_model.py
    features = np.array([[
        applicant.income, 
        applicant.credit_score, 
        applicant.loan_amount, 
        applicant.employment_years
    ]])
    
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    
    return {"approved": bool(prediction[0])}