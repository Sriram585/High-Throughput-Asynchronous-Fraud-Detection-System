# High-Throughput Asynchronous Fraud Detection & XAI Pipeline

An end-to-end, event-driven machine learning inference microservice designed for low-latency financial transaction processing under extreme class imbalance, featuring out-of-band SHAP (SHapley Additive exPlanations) attribution.

---

## 1. Architectural Bottlenecks & Engineering Solutions

Deploying tree-based Explainable AI models (`xgboost` + `shap.TreeExplainer`) inside synchronous web APIs introduces three severe production failure modes:
1. **Event-Loop Starvation**: Computing exact Shapley values is CPU-bound. Running `TreeExplainer` inside a synchronous FastAPI route blocks the Python GIL, inflating latency beyond payment gateway SLAs (<50ms) during traffic spikes.
2. **OS Path & Database Lock Collisions**: Coupling Linux Docker containers directly to local Windows MLflow SQLite tracking databases (`mlruns.db`) triggers cross-platform URI crashes and file-level read/write locks.
3. **OpenMP Multiprocessing Deadlocks**: Default Celery `prefork` pools (`os.fork()`) silently deadlock when duplicating worker processes that have pre-initialized C++ OpenMP thread pools inside XGBoost and SHAP.

### System Design Resolution
* **Asynchronous Ingestion Gateway (`main.py`)**: FastAPI validates incoming payloads via strict Pydantic v2 schemas, enqueues tasks into a Redis broker, and returns an HTTP `202 Accepted` with a tracking `task_id` in <10ms.
* **Isolated XAI Worker Nodes (`worker.py`)**: Dedicated Celery workers running under non-forking `--pool=solo` execution pull tasks from Redis, evaluate transactions against pre-warmed XGBoost and SHAP objects cached in RAM, and write human-readable feature attributions back to the Redis state store.
* **Leakage-Free Training & Artifact Decoupling (`train.py`)**: Trains on 6.3M PaySim financial records after stripping post-cancellation target-leakage columns (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`). Counteracts the 0.13% minority class imbalance via `scale_pos_weight` and exports an OS-independent serving artifact (`./model_dir/`).

---

## 2. System Architecture

```text
  [ Client Request ]
          │ (POST /predict)
          ▼
┌──────────────────┐
│   FastAPI Edge   │ ──── Validates schema (Pydantic v2) & returns HTTP 202 + `task_id` (<10ms)
└──────────────────┘
          │ Enqueues JSON payload
          ▼
┌──────────────────┐
│   Redis Broker   │ (In-Memory Message Queue & State Backend)
└──────────────────┘
          │ Consumes task asynchronously
          ▼
┌──────────────────┐
│  Celery Worker   │ ──── Pre-warmed XGBoost + SHAP TreeExplainer (Loaded from ./model_dir/)
└──────────────────┘      (Runs with --pool=solo to prevent C++ OpenMP fork deadlocks)
          │ Writes probability + Top 3 SHAP risk factors
          ▼
┌──────────────────┐
│ Redis State Store│ ◄─── Polled via GET /status/{task_id}
└──────────────────┘
```

---

## 3. Repository Structure

```text
├── data/                   # Raw 6.3M-row PaySim CSV (excluded via .gitignore / .dockerignore)
├── model_dir/              # Standalone, OS-independent MLflow model artifact for Docker
├── train.py                # Preprocessing, leakage removal, XGBoost training, MLflow logging
├── main.py                 # FastAPI asynchronous ingestion gateway and status polling routes
├── worker.py               # Celery task definition and in-memory XGBoost/SHAP execution
├── docker-compose.yml      # Multi-container orchestration (FastAPI, Celery Worker, Redis 7)
├── Dockerfile              # Production container image definition (python:3.11-slim)
├── requirements.txt        # Locked production dependencies
├── .dockerignore           # Excludes 470MB dataset and local SQLite logs from build context
└── .gitignore              # Prevents dataset and local tracking artifacts from entering Git
```

---

## 4. Deployment & Verification

### Step 1: Launch the Distributed Cluster
Ensure `./model_dir/` is exported via `train.py`, then build and start the 3-node cluster:
```bash
docker compose up --build -d
```

### Step 2: Enqueue a Transaction (`POST /predict`)
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "step": 1,
  "amount": 95000.50,
  "type_CASH_IN": 0,
  "type_CASH_OUT": 0,
  "type_DEBIT": 0,
  "type_PAYMENT": 0,
  "type_TRANSFER": 1
}'
```

**Immediate Response (`202 Accepted`):**
```json
{
  "task_id": "dc2b2913-8c09-430c-9164-1276711c73a1",
  "status": "PROCESSING"
}
```

### Step 3: Poll Inference & XAI Result (`GET /status/{task_id}`)
```bash
curl -X 'GET' 'http://localhost:8000/status/dc2b2913-8c09-430c-9164-1276711c73a1'
```

**Completed Worker Output:**
```json
{
  "task_id": "dc2b2913-8c09-430c-9164-1276711c73a1",
  "status": "COMPLETED",
  "result": {
    "fraud_probability": 0.4991,
    "risk_classification": "LEGITIMATE",
    "top_risk_factors": {
      "step": 0.7055,
      "type_CASH_OUT": -0.6909,
      "type_TRANSFER": -0.5003
    }
  }
}
```