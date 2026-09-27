# Use a lightweight official Python image
FROM python:3.11-slim

# Prevent Python from writing temporary .pyc files and force standard output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code and the serialized machine learning model
COPY src/ ./src/
COPY models/ ./models/

# Expose the port that FastAPI runs on
EXPOSE 8000

# Command to boot up the Uvicorn web server when the container starts
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]