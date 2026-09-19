import streamlit as st
import pandas as pd
import joblib


model = joblib.load("employee_promotion_model.pkl")

st.title("Employee Promotion Prediction")


st.sidebar.header("Enter Employee Details")

department = st.sidebar.selectbox("Department", 
    ["Sales & Marketing", "Operations", "HR", "Finance", "Technology", "Procurement"])
education = st.sidebar.selectbox("Education", ["Bachelor's", "Master's", "PhD", "Other"])
gender = st.sidebar.radio("Gender", ["Male", "Female"])
recruitment_channel = st.sidebar.selectbox("Recruitment Channel", ["Sourcing", "Other", "Referred"])
awards_won = st.sidebar.radio("Awards Won?", [0, 1])
previous_year_rating = st.sidebar.slider("Previous Year Rating", 0, 5, 3)
avg_training_score = st.sidebar.slider("Avg Training Score", 0, 100, 50)


input_data = pd.DataFrame({
    "department": [department],
    "education": [education],
    "gender": [gender],
    "recruitment_channel": [recruitment_channel],
    "awards_won": [awards_won],
    "previous_year_rating": [previous_year_rating],
    "avg_training_score": [avg_training_score]
})


if st.sidebar.button("Predict Promotion"):
    prediction = model.predict(input_data)
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success("Employee is likely to be promoted!")
    else:
        st.warning("Employee is not likely to be promoted.")
