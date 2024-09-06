import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object  # Importing save_object function

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")  # Add path for preprocessor.pkl

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset
            df = pd.read_csv('notebook\data\sampled_data.csv')
            logging.info('Read the dataset as dataframe')

            # Preprocessing steps as before
            df.replace('', np.nan, inplace=True)
            numerical_columns = ['age', 'bmi', 'children', 'charges']
            df[numerical_columns] = df[numerical_columns].astype(float)
            categorical_columns = ['gender', 'smoker', 'region', 'medical_history', 'family_medical_history',
                                   'exercise_frequency', 'occupation', 'coverage_level']
            df[categorical_columns] = df[categorical_columns].astype('category')

            df['age'] = df['age'].fillna(df['age'].median())
            df['gender'] = df['gender'].fillna(df['gender'].mode()[0])
            df['children'] = df['children'].fillna(round(df['children'].median()))
            df['occupation'] = df['occupation'].fillna(df['occupation'].mode()[0])

            for col in ['medical_history', 'family_medical_history']:
                if 'No history' not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories('No history')
                df[col] = df[col].fillna('No history')

            X = df.drop('charges', axis=1)
            y = df['charges']

            categorical_features = ['gender', 'smoker', 'region', 'medical_history', 'family_medical_history',
                                    'exercise_frequency', 'occupation', 'coverage_level']
            numerical_features = ['age', 'bmi', 'children']

            # Create and fit preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', 'passthrough', numerical_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ]
            )
            preprocessor.fit(X)  # Fit preprocessor on the data

            # Save the preprocessor object using save_object
            save_object(
                file_path=self.ingestion_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            # Encoding steps and correlation filtering as before...
            X_encoded = preprocessor.transform(X)
            encoded_feature_names = (numerical_features +
                                     list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))
            encoded_df = pd.DataFrame(X_encoded, columns=encoded_feature_names)
            encoded_df['charges'] = y.values

            correlation_matrix = encoded_df.corr()
            df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
            correlation_matrix = df_encoded.corr()
            target_variable = 'charges'
            threshold = 0.05
            relevant_features = correlation_matrix[target_variable][(correlation_matrix[target_variable] > threshold) |
                                                                    (correlation_matrix[target_variable] < -threshold)].index
            df_relevant = df_encoded[relevant_features]
            if target_variable not in df_relevant.columns:
                df_relevant[target_variable] = df_encoded[target_variable]

            Q1 = df['charges'].quantile(0.25)
            Q3 = df['charges'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df_no_outliers = df_relevant[(df_relevant['charges'] >= lower_bound) &
                                         (df_relevant['charges'] <= upper_bound)]

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df_no_outliers.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df_no_outliers, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.preprocessor_obj_file_path  # Return the path to the preprocessor object
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # Data Ingestion
        print("Starting data ingestion...")
        obj = DataIngestion()
        train_data, test_data, preprocessor_path = obj.initiate_data_ingestion()
        print("Data ingestion completed.")

        # Data Transformation (consider running this in batches or using GPU-accelerated methods)
        print("Starting data transformation...")
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
        print("Data transformation completed.")

        # Free up memory by explicitly deleting unnecessary objects and calling garbage collection
        del train_data, test_data
        import gc
        gc.collect()

        # Model Training
        print("Starting model training...")
        modeltrainer = ModelTrainer()

        # Optionally, you can use checkpointing or batch model training if needed
        r2 = modeltrainer.initiate_model_trainer(train_arr, test_arr)
        print(f"Model training completed with R2 score: {r2}")



    except CustomException as e:
        print(f"An error occurred: {e}")

