import os
import shutil
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

# 1. Setup MLflow Tracking
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("paysim_fraud_xai")

def train_paysim():
    print("Loading PaySim dataset...")
    df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv") 

    # 2. Data Preprocessing & Leakage Prevention
    drop_cols = [
        "nameOrig", "nameDest", "isFlaggedFraud",
        "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"
    ]
    df = df.drop(columns=drop_cols)

    # One-Hot Encode 'type'
    df = pd.get_dummies(df, columns=["type"], dtype=int)

    # Separate features and target
    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 3. Start MLflow Run
    with mlflow.start_run():
        imbalance_ratio = float(y_train.value_counts()[0] / y_train.value_counts()[1])

        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "max_depth": 5,
            "learning_rate": 0.1,
            "scale_pos_weight": imbalance_ratio
        }
        mlflow.log_params(params)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        print("Training XGBoost Model...")
        model = xgb.train(params, dtrain, num_boost_round=50)

        # Evaluate Model
        preds = model.predict(dtest)
        pr_auc = average_precision_score(y_test, preds)
        mlflow.log_metric("pr_auc", pr_auc)

        # 4. Register the Model in MLflow
        mlflow.xgboost.log_model(
            xgb_model=model, 
            artifact_path="fraud_model",
            registered_model_name="XGBoost_PaySim_XAI"
        )
        
        # 5. Export clean, path-independent artifact for Docker
        export_path = "./model_dir"
        if os.path.exists(export_path):
            shutil.rmtree(export_path)
        mlflow.xgboost.save_model(xgb_model=model, path=export_path)
        
        print(f"Run completed. PR AUC: {pr_auc:.4f}")
        print(f"Clean serving model exported to {export_path}")

if __name__ == "__main__":
    train_paysim()