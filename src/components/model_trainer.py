import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module="xgboost")

import os
import sys
from dataclasses import dataclass

import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            # Define the model
            xgb_model = XGBRegressor(objective='reg:squarederror', tree_method='gpu_hist', random_state=42)
            
            # Define the parameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9]
            }
            
            # Grid search
            grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            
            # Get the best model
            best_xgb_model = grid_search.best_estimator_

            logging.info(f"Best XGBoost model: {best_xgb_model}")
            
            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_xgb_model
            )

            # Make predictions
            y_pred = best_xgb_model.predict(X_test)

            # Evaluate the model
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

            logging.info(f"Model evaluation metrics: RMSE={rmse}, MAE={mae}, R2={r2}, MAPE={mape}")

            return r2
        
        except Exception as e:
            raise CustomException(e, sys)
