import streamlit as st
import requests

# Hide Streamlit elements
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            .stDeployButton {display:none;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Loan Approval Predictor")
st.write("Enter applicant details:")

# Updated inputs to match your dataset columns
income = st.number_input("Annual Income", min_value=0, value=50000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
loan_amount = st.number_input("Loan Amount", min_value=0, value=15000)
employment_years = st.number_input("Employment History (Years)", min_value=0, value=2)

if st.button("Predict"):
    payload = {
        "income": income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "employment_years": employment_years
    }
    
    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if result["approved"]:
                st.success("Status: Approved")
            else:
                st.error("Status: Denied")
        else:
            st.write("Error: Could not get a prediction. Check backend terminal for details.")
    except Exception:
        st.write("Error: Backend server is not running.")