from fastapi import FastAPI, status
from pydantic import BaseModel
from celery.result import AsyncResult
from worker import celery_app, run_fraud_inference

app = FastAPI(title="Asynchronous Explainable Fraud Detection API")

class TransactionRequest(BaseModel):
    step: int
    amount: float
    type_CASH_IN: int
    type_CASH_OUT: int
    type_DEBIT: int
    type_PAYMENT: int
    type_TRANSFER: int

@app.post("/predict", status_code=status.HTTP_202_ACCEPTED)
def enqueue_prediction(transaction: TransactionRequest):
    # Enqueue task to Redis broker asynchronously (.delay)
    task = run_fraud_inference.delay(transaction.model_dump())
    return {
        "task_id": task.id,
        "status": "PROCESSING"
    }

@app.get("/status/{task_id}")
def get_task_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)
    
    if task_result.state == "PENDING":
        return {"task_id": task_id, "status": "PENDING"}
    
    elif task_result.state == "SUCCESS":
        return {
            "task_id": task_id,
            "status": "COMPLETED",
            "result": task_result.result
        }
    
    elif task_result.state == "FAILURE":
        return {
            "task_id": task_id,
            "status": "FAILED",
            "error": str(task_result.info)
        }
    
    return {"task_id": task_id, "status": task_result.state}