import streamlit as st
import requests
import threading
import uvicorn
import socket
import time
from main import app


def is_port_in_use(port: int = 8000) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


@st.cache_resource
def start_backend_server():
    if not is_port_in_use(8000):
        def _run_server():
            uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

        server_thread = threading.Thread(target=_run_server, daemon=True)
        server_thread.start()
        
        # Give server time to bind and start listening
        for _ in range(30):
            if is_port_in_use(8000):
                break
            time.sleep(0.1)
    return True


start_backend_server()

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
    
    approved = None
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=3)
        if response.status_code == 200:
            result = response.json()
            approved = result.get("approved")
    except Exception:
        # Fallback to direct model inference if local HTTP is restricted in cloud containers
        try:
            from main import model, scaler
            import pandas as pd
            features = pd.DataFrame([payload])
            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)
            approved = bool(prediction[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    if approved is not None:
        if approved:
            st.success("Status: Approved")
        else:
            st.error("Status: Denied")
    else:
        st.error("Could not obtain a prediction. Please check server logs.")