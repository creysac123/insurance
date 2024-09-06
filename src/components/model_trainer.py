import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
            logging.info("Starting model training process...")

            # Split training and test input data
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            # Scale the data
            logging.info("Scaling the data using StandardScaler")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Define the parameter grid for HuberRegressor
            param_grid = {
                'epsilon': [1.1, 1.35, 1.5],  # Epsilon parameter to control robustness
                'alpha': [0.0001, 0.001, 0.01],  # Regularization parameter
                'max_iter': [1000, 1500, 2000],  # Maximum number of iterations
                'tol': [1e-4, 1e-5, 1e-6]  # Tolerance for stopping criteria
            }

            # Initialize HuberRegressor
            logging.info("Initializing HuberRegressor model")
            huber_model = HuberRegressor()

            # Initialize grid search
            logging.info("Initializing GridSearchCV for hyperparameter tuning")
            grid_search = GridSearchCV(estimator=huber_model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error', verbose=2)

            # Start model training
            logging.info("Fitting the HuberRegressor model on the training data")
            grid_search.fit(X_train_scaled, y_train)

            logging.info("GridSearchCV fitting complete")

            # Get the best model
            best_huber_model = grid_search.best_estimator_
            logging.info(f"Best HuberRegressor model: {best_huber_model}")
            
            # Save the best model
            logging.info("Saving the best model to the specified file path")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_huber_model
            )

            # Make predictions
            logging.info("Making predictions on the test data")
            y_pred = best_huber_model.predict(X_test_scaled)

            # Evaluate the model
            logging.info("Evaluating the model with RMSE, MAE, R², and MAPE")
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

            logging.info(f"Model evaluation metrics: RMSE={rmse}, MAE={mae}, R2={r2}, MAPE={mape}")

            return r2
        
        except Exception as e:
            logging.error(f"An error occurred during model training: {str(e)}")
            raise CustomException(e, sys)
