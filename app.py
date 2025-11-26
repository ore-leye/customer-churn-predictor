import streamlit as st
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Churn Predictor", page_icon="warning", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.pkl")
    columns = joblib.load("model_columns.pkl")
    explainer = joblib.load("explainer.pkl")
    return model, columns, explainer

model, columns, explainer = load_artifacts()

st.title("Customer Churn Predictor")
st.markdown("### Will this customer leave your bank/fintech? Get instant risk score + reasons")

col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 24)
    monthly = st.number_input("Monthly Charges (₦)", 0, 200000, 50000)
    total = st.number_input("Total Charges (₦)", 0, 10000000, 1000000)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

with col2:
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])

if st.button("Predict Churn Risk", type="primary"):
    with st.spinner("Analyzing customer behavior..."):
        input_data = pd.DataFrame([{
            'SeniorCitizen': 1 if senior == "Yes" else 0,
            'tenure': tenure,
            'MonthlyCharges': monthly,
            'TotalCharges': total,
            'gender_Male': 1,  # dummy
            'Partner_Yes': 1,
            'Dependents_Yes': 0,
            'PhoneService_Yes': 1,
            'Contract_One year': 1 if contract == "One year" else 0,
            'Contract_Two year': 1 if contract == "Two year" else 0,
            'PaperlessBilling_Yes': 1 if paperless == "Yes" else 0,
            'PaymentMethod_Electronic check': 1 if payment == "Electronic check" else 0,
            'InternetService_Fiber optic': 1 if internet == "Fiber optic" else 0,
            'InternetService_No': 1 if internet == "No" else 0,
            'OnlineSecurity_Yes': 1 if online_sec == "Yes" else 0,
            'TechSupport_Yes': 1 if tech_support == "Yes" else 0,
        }])
        
        input_encoded = input_data.reindex(columns=columns, fill_value=0)
        prob = model.predict_proba(input_encoded)[0][1]
        prediction = model.predict(input_encoded)[0]

    st.markdown(f"## {'High Risk – Likely to Churn' if prediction else 'Low Risk – Likely to Stay'}")
    st.metric("Churn Probability", f"{prob:.1%}")
    
    if prediction:
        st.error("This customer is likely to leave soon!")
    else:
        st.success("This customer is loyal and safe")

    # SHAP explanation
    shap_value = explainer.shap_values(input_encoded)[0]
    top_features = pd.Series(shap_value, index=columns).abs().sort_values(ascending=False).head(5)
    st.bar_chart(top_features)
    st.caption("Top 5 reasons driving this prediction (SHAP values)")

st.markdown("---")
st.caption("Built by Ore Oyeleye | XGBoost + SHAP | Deployed with Streamlit")