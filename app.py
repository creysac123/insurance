from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for the home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Get form data and create a CustomData instance
        data = CustomData(
            age=float(request.form.get('age')),
            gender=request.form.get('gender'),
            bmi=float(request.form.get('bmi')),
            children=int(request.form.get('children')),
            smoker=request.form.get('smoker'),
            region=request.form.get('region'),
            medical_history=request.form.get('medical_history'),
            family_medical_history=request.form.get('family_medical_history'),
            exercise_frequency=request.form.get('exercise_frequency'),
            occupation=request.form.get('occupation'),
            coverage_level=request.form.get('coverage_level')
        )

        # Convert the form data to a DataFrame
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        # Create a PredictPipeline instance and make a prediction
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        # Return the results in the home page
        return render_template('home.html', results=f'Estimated Insurance Charges: ${results[0]:.2f}')

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=80)
