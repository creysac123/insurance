import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Title of the application
st.title("Insurance Charges Prediction")

# Sidebar form for user input
with st.form(key='insurance_form'):
    st.header("Insurance Charge Prediction Form")

    # Input fields
    gender = st.radio("Gender", options=["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=120, step=1, format="%d", placeholder="Enter your age")
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1, format="%.1f", placeholder="Enter your BMI")
    children = st.number_input("Number of Children", min_value=0, max_value=10, step=1, format="%d", placeholder="Enter number of children")
    smoker = st.radio("Smoker", options=["Yes", "No"])
    region = st.radio("Region", options=["Northeast", "Northwest", "Southeast", "Southwest"])
    exercise_frequency = st.radio("Exercise Frequency", options=["Never", "Rarely", "Occasionally", "Frequently"])
    occupation = st.radio("Occupation", options=["Blue collar", "Student", "White collar", "Unemployed"])
    coverage_level = st.radio("Coverage Level", options=["Basic", "Standard", "Premium"])
    medical_history = st.radio("Medical History", options=["No history", "Diabetes", "High blood pressure", "Heart disease"])
    family_medical_history = st.radio("Family Medical History", options=["No history", "Diabetes", "High blood pressure", "Heart disease"])

    # Submit button
    submit_button = st.form_submit_button(label="Predict Insurance Charges")

# On form submission, process data and predict
if submit_button:
    # Create a CustomData instance
    data = CustomData(
        age=age,
        gender=gender,
        bmi=bmi,
        children=children,
        smoker=smoker,
        region=region,
        medical_history=medical_history,
        family_medical_history=family_medical_history,
        exercise_frequency=exercise_frequency,
        occupation=occupation,
        coverage_level=coverage_level
    )

    # Convert data into a DataFrame
    pred_df = data.get_data_as_data_frame()

    # Display the input data
    st.subheader("Input Data")
    st.write(pred_df)

    # Create a PredictPipeline instance
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)

    # Display the prediction results
    st.success(f"The predicted insurance charges are: ${results[0]:.2f}")
