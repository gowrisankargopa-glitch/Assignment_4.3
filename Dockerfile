# Use an official Python image as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies first
RUN pip install --no-cache-dir joblib pandas scikit-learn pytz redis fastapi uvicorn sqlalchemy

# Copy the .whl package (includes all Python files and model files)
COPY dist/house_price_prediction-0.1.0-py3-none-any.whl .
RUN pip install --no-cache-dir house_price_prediction-0.1.0-py3-none-any.whl

# Set environment variable to ensure Python output is sent straight to terminal
ENV PYTHONUNBUFFERED=1

# Command to run the Python script when the container starts
CMD ["python3", "-m", "house_price_prediction.main"]