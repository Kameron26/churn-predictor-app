
import streamlit as st
import joblib
import pandas as pd

#Title
st.title("Customer Churn Predictor")

#Load saved model
model = joblib.load("churn_model.pkl")

#Collect user input (as form or sidebar)
st.header("Customer Details")
gender = st.selectbox("Gender", ["Male","Female"])
senior = st.selectbox("Senior Citizen", ["Yes","No"])
partner = st.selectbox("Has Partner?", ["Yes","No"])
dependents = st.selectbox("Has Dependents?", ["Yes","No"])
tenure = st.slider("Tenure (months)",0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", min_value = 0.0)
total_charges = st.number_input("Total Charges", min_value = 0.0)

input_dict = {"gender":[gender], "SeniorCitizen": [1 if senior == "Yes" else 0], "Partner":[partner], "Dependents":[dependents],"tenure":[tenure],
              "MonthlyCharges":[monthly_charges], "TotalCharges": [total_charges]}
input_df = pd.DataFrame(input_dict)

#Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    if prediction == 1:
        st.error(f"High Risk of Churn - {round(probability*100,2)}%")
    else:
        st.success(f"Customer likely to stay - {round((100-probability)*100,2)}%")