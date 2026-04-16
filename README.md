# COMP 3610 Assignment 4: MLOps and Model Deployment

**Name:** Varsha Roopchand  
**Student ID:** 816039243  
**Course:** COMP 3610 Big Data Analytics

## Project Description

This project deploys a machine learning model for predicting NYC Yellow Taxi `tip_amount` values using the dataset and feature engineering approach developed earlier in the course. The assignment covers the full deployment workflow: experiment tracking with MLflow, model serving with FastAPI, automated API testing with pytest, and containerization with Docker and Docker Compose.

The final deployed model is exposed through a REST API that supports:

- single prediction requests
- batch prediction requests
- health monitoring
- model metadata inspection

## Repository Contents

This repository includes the files required by the Assignment 4 specification:

- `assignment4.ipynb` - notebook documenting MLflow experiments, API development, testing, and Docker deployment
- `app.py` - FastAPI application source code
- `model_utils.py` - helper functions for feature engineering, training, evaluation, and artifact saving
- `train_and_log.py` - script for MLflow experiment logging and model registration
- `test_app.py` - pytest test suite for the API
- `Dockerfile` - Docker image definition for the prediction service
- `docker-compose.yml` - Docker Compose orchestration file
- `requirements.txt` - pinned Python dependencies
- `README.md` - setup instructions and project description
- `.gitignore` - excludes large or generated files
- `.dockerignore` - excludes unnecessary Docker build context
- `submission.txt` - submission template with repository and commit details

## Prerequisites

To run this project successfully, the following are required:

- Python 3.12
- Docker Desktop
- PowerShell or another terminal

## Python Dependencies

The main dependencies used in this project are:

mlflow 3.11.1
fastapi 0.135.3
uvicorn 0.42.0
pydantic 2.12.5
httpx 0.28.1
scikit-learn 1.8.0
joblib 1.5.3
pandas 2.3.3
numpy 2.4.1
pytest 9.0.3
pyarrow 23.0.0
requests 2.32.5

AThis is also available in `requirements.txt`.

## Setup Instructions

Install the required Python packages:

```powershell
pip install -r requirements.txt 
```

## Part 1: MLflow Experiment Tracking

Start the MLflow UI:

```powershell
mlflow ui --port 5000
```

Open the MLflow dashboard in the browser:

```text
http://127.0.0.1:5000
```

Train and log the models:

```powershell
py train_and_log.py
```

This script:

- downloads the dataset if it is missing
- reuses the Assignment 2-style feature engineering pipeline
- trains and compares at least two regression models
- logs parameters, metrics, tags, and model artifacts to MLflow
- registers the best model
- saves deployment artifacts into `models/`

## Part 2: FastAPI Model Serving

Start the FastAPI application locally:

```powershell
uvicorn app:app --reload --port 8000
```

Open the Swagger UI:

```text
http://127.0.0.1:8000/docs
```
and
```text
http://localhost:8000/
```
The API includes the required endpoints:

- `POST /predict`
- `POST /predict/batch`
- `GET /health`
- `GET /model/info`

### Example Single Prediction Request

```json
{
  "pickup_hour": 10,
  "pickup_day_of_week": 3,
  "is_weekend": false,
  "trip_duration_minutes": 15.0,
  "trip_speed_mph": 13.0,
  "log_trip_distance": 1.2,
  "fare_per_mile": 3.5,
  "fare_per_minute": 0.8,
  "pickup_borough": "Manhattan",
  "dropoff_borough": "Brooklyn",
  "passenger_count": 2,
  "trip_distance": 3.8,
  "fare_amount": 13.4
}
```

## Part 2 Testing

Run the automated API tests:

```powershell
cd "C:\Users\earth\OneDrive\Documents\New project\assignment_4"
py -m pytest -q
```

The tests cover:

- successful single prediction
- successful batch prediction
- invalid input rejection
- health check verification
- model info verification
- zero-distance edge case
- batch size limit enforcement

## Part 3: Docker Image Build

Build the Docker image:

```powershell
docker version
docker pull python:3.11-slim
docker build -t taxi-tip-api .
```

The custom image built for this project was approximately **1.7 GB**.

Run the container:

```powershell
docker run -d -p 8000:8000 --name taxi-tip-api-container taxi-tip-api
```

Verify the API from outside the container:

```powershell
curl.exe http://localhost:8000/health
```

Stop and remove the standalone container:

```powershell
docker stop taxi-tip-api-container
docker rm taxi-tip-api-container
```

## Part 3: Docker Compose Deployment

Start the service with Docker Compose:

```powershell
docker compose up -d
```

Check the service status:

```powershell
docker compose ps
```

Example prediction requests from PowerShell:

```powershell
$body = @{
  pickup_hour = 18
  pickup_day_of_week = 5
  is_weekend = $true
  trip_duration_minutes = 22.0
  trip_speed_mph = 14.5
  log_trip_distance = 1.6
  fare_per_mile = 4.0
  fare_per_minute = 0.95
  pickup_borough = "Queens"
  dropoff_borough = "Manhattan"
  passenger_count = 1
  trip_distance = 5.2
  fare_amount = 20.9
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $body
```

Shut the service down cleanly:

```powershell
docker compose down
```

## Container Configuration

The Docker Compose setup uses the following configuration:

- host port `8000` mapped to container port `8000`
- `MODEL_PATH=/app/models/taxi_tip_model.joblib`
- `MODEL_INFO_PATH=/app/models/model_info.json`

The following model artifacts must exist before running the containerized API:

- `models/taxi_tip_model.joblib`
- `models/model_info.json`

## Important Notes

- Screenshots for MLflow, Swagger UI, and Docker workflow are stored in `screenshots/`.

## AI ASSISTANCE DISCLOSURE
AI (Chat GPT) was used in the process of creating this assignment for help understanding the project requirements, debugging, understanding the results and with the structure of code. All AI generated code was understood before submission.
