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
    age = st.text_input("Age", placeholder="Enter your age")
    bmi = st.text_input("BMI", placeholder="Enter your BMI (e.g., 25.5)")
    children = st.text_input("Number of Children", placeholder="Enter number of children")
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
    try:
        # Ensure numeric inputs are valid
        age = float(age)
        bmi = float(bmi)
        children = int(children)

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

    except ValueError:
        st.error("Please enter valid numeric values for Age, BMI, and Number of Children.")
