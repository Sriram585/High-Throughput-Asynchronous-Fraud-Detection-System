import os
import pandas as pd
import xgboost as xgb
import shap
import mlflow.xgboost
from celery import Celery

# Configure Redis connection URL from environment variables
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "fraud_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL
)

# Enforce JSON serialization and fail-fast timeouts
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    result_expires=3600,  # Purge task results from Redis RAM after 1 hour
    worker_prefetch_multiplier=1  # Prevent worker starvation on uneven CPU loads
)

FEATURE_NAMES = [
    'step', 'amount', 'type_CASH_IN', 'type_CASH_OUT', 
    'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
]

# Load Champion Model and SHAP Explainer into worker RAM at startup
try:
    model_uri = "./model_dir"
    model = mlflow.xgboost.load_model(model_uri)
    explainer = shap.TreeExplainer(model)
    print("Worker initialized: Champion model and SHAP explainer cached in memory.")
except Exception as e:
    model = None
    explainer = None
    print(f"CRITICAL: Worker failed to load model. Error: {e}")

@celery_app.task(name="run_fraud_inference")
def run_fraud_inference(payload: dict):
    if model is None or explainer is None:
        raise RuntimeError("Model weights are not loaded in worker memory.")
    
    # Reconstruct exact feature ordering
    input_data = pd.DataFrame([payload])[FEATURE_NAMES]
    
    # Execute XGBoost inference
    dmatrix = xgb.DMatrix(input_data)
    fraud_probability = float(model.predict(dmatrix)[0])
    
    # Compute SHAP attributions
    shap_values = explainer.shap_values(input_data)[0]
    feature_contributions = sorted(
        zip(FEATURE_NAMES, shap_values), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )
    
    top_factors = {
        feat: round(float(contribution), 4) 
        for feat, contribution in feature_contributions[:3]
    }
    
    return {
        "fraud_probability": round(fraud_probability, 4),
        "risk_classification": "FRAUD" if fraud_probability > 0.5 else "LEGITIMATE",
        "top_risk_factors": top_factors
    }