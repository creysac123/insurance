import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")

            # Preprocess the input features
            data_scaled = preprocessor.transform(features)

            # Make predictions
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, 
                 age: float, 
                 gender: str, 
                 bmi: float, 
                 children: int, 
                 smoker: str, 
                 region: str, 
                 medical_history: str, 
                 family_medical_history: str, 
                 exercise_frequency: str, 
                 occupation: str, 
                 coverage_level: str):
        
        self.age = age
        self.gender = gender
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = region
        self.medical_history = medical_history
        self.family_medical_history = family_medical_history
        self.exercise_frequency = exercise_frequency
        self.occupation = occupation
        self.coverage_level = coverage_level

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "gender": [self.gender],
                "bmi": [self.bmi],
                "children": [self.children],
                "smoker": [self.smoker],
                "region": [self.region],
                "medical_history": [self.medical_history],
                "family_medical_history": [self.family_medical_history],
                "exercise_frequency": [self.exercise_frequency],
                "occupation": [self.occupation],
                "coverage_level": [self.coverage_level],
            }

            # Convert the dictionary to a DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
