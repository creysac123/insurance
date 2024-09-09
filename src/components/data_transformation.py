import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    # Path where the preprocessor object is stored
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # At this point, the preprocessing has already been done in data ingestion
            # If no further transformation is needed, simply move on to the next steps

            target_column_name = "charges"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Potential additional steps:
            # - Outlier detection and removal
            # - Feature scaling or normalization
            # - Feature engineering or creation of interaction features

            # Prepare arrays for model training
            train_arr = np.c_[input_feature_train_df, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_df, np.array(target_feature_test_df)]

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
