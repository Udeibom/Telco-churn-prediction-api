# Telco Customer Churn Prediction API

This is a FastAPI-based service for predicting customer churn using a trained ML model.

## 🚀 Run Locally (without Docker)

1. Install dependencies:
   pip install -r requirements.txt

2. Start API:
   uvicorn main:app --reload

3. Open in browser:
   http://127.0.0.1:8000/docs

---

## 🐳 Run with Docker

1. Build the image:
   docker build -t churn-api .

2. Run the container:
   docker run -p 8000:8000 churn-api

3. Test:
   http://127.0.0.1:8000/docs

---

## 📌 Endpoints

- `/health` → health check  
- `/predict` → churn prediction (POST JSON)  

Sample request in `sample_request.json`.
