# main.py
import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
import shap
from huggingface_hub import hf_hub_download, login

# Constants for Hugging Face repo and model filename
HUGGINGFACE_REPO = "https://huggingface.co/Udeibom/telco_churn_model"
MODEL_FILENAME = "best_baseline_model.pkl"

def download_model():
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        login(token=token)  # Authenticate to access private repo
    model_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename=MODEL_FILENAME)
    return model_path

# Load model: either from MODEL_PATH env var or download from Hugging Face
MODEL_PATH = os.getenv("MODEL_PATH")
if MODEL_PATH is None:
    try:
        MODEL_PATH = download_model()
    except Exception as e:
        raise RuntimeError(f"Could not download model from Hugging Face: {e}")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load model at {MODEL_PATH}: {e}")

# Try to extract pipeline parts
preprocessor = None
classifier = model
if hasattr(model, "named_steps"):
    preprocessor = model.named_steps.get("preprocessor", None)
    classifier = model.named_steps.get("classifier", model)

# Build SHAP explainer lazily (for tree-based models)
explainer = None
try:
    if "xgb" in str(type(classifier)).lower() or hasattr(classifier, "feature_importances_"):
        explainer = shap.TreeExplainer(classifier)  # may be slow on first call
except Exception:
    explainer = None

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predict churn and return explanations (small SHAP summary)",
    version="1.1"
)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Strict input schema
class CustomerData(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    MonthlyCharges: float
    TotalCharges: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "model_type": str(type(classifier)),
        "features": list(preprocessor.get_feature_names_out()) if preprocessor else "unknown",
        "explainability": (
            "shap" if explainer
            else "feature_importances" if hasattr(classifier, "feature_importances_")
            else "none"
        )
    }

@app.post("/predict")
def predict(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    try:
        proba = float(model.predict_proba(df)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    label = "Yes" if proba >= 0.5 else "No"
    return {"churn_probability": round(proba, 4), "churn_label": label}

@app.post("/predict_with_explain")
def predict_with_explain(data: CustomerData, top_k: int = 5):
    df = pd.DataFrame([data.dict()])
    try:
        proba = float(model.predict_proba(df)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    label = "Yes" if proba >= 0.5 else "No"

    explanation = None
    if explainer is not None and preprocessor is not None:
        try:
            X_trans = preprocessor.transform(df)
            try:
                s = explainer(X_trans)
                vals = s.values if hasattr(s, "values") else np.array(s)
            except Exception:
                vals = explainer.shap_values(X_trans)

            shap_arr = np.array(vals[1]) if isinstance(vals, list) and len(vals) > 1 else np.array(vals[0] if isinstance(vals, list) else vals)
            contribs = shap_arr[0]

            try:
                feat_names = preprocessor.get_feature_names_out()
            except Exception:
                feat_names = [f"f{i}" for i in range(contribs.shape[0])]

            idx = np.argsort(np.abs(contribs))[::-1][:top_k]
            explanation = [{"feature": feat_names[i], "shap_value": float(contribs[i])} for i in idx]
        except Exception as e:
            explanation = [{"error": "explanation_failed", "detail": str(e)}]

    elif hasattr(classifier, "feature_importances_"):
        try:
            fi = classifier.feature_importances_
            feat_names = (
                list(preprocessor.get_feature_names_out())
                if preprocessor is not None
                else [f"f{i}" for i in range(len(fi))]
            )
            idx = np.argsort(fi)[::-1][:top_k]
            explanation = [{"feature": feat_names[i], "importance": float(fi[i])} for i in idx]
        except Exception as e:
            explanation = [{"error": "importance_failed", "detail": str(e)}]

    return {"churn_probability": round(proba, 4), "churn_label": label, "explanation": explanation}
