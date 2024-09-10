# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /insurance

# Copy only the necessary files
COPY artifacts/model.pkl /insurance/artifacts/
COPY artifacts/preprocessor.pkl /insurance/artifacts/

# Copy the predict pipeline from src/pipeline
COPY src/pipeline/predict_pipeline.py /insurance/src/pipeline/

# Copy app.py (or your entry point file)
COPY app.py /insurance/

# Copy the templates folder
COPY templates /insurance/templates

# Copy utils.py, logger.py, and exception.py from src folder
COPY src/utils.py /insurance/src/
COPY src/logger.py /insurance/src/
COPY src/exception.py /insurance/src/

# Copy requirements.txt
COPY requirements.txt /insurance/

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 80 

# Command to run the application
CMD ["python", "app.py"]
