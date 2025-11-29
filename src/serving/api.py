import joblib
import pandas as pd
import os
import io
import glob
import re
import time
import logging
import shap
from datetime import datetime
from dotenv import load_dotenv
from src.monitoring.init_db import init_db

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Depends, HTTPException, Security, Request
from fastapi.requests import Request
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from fastapi.responses import FileResponse


from src.monitoring.logger import log_prediction
from src.monitoring.drift import check_drift
from src.retraining.retrain import main as retrain_model

# Load environment variables
load_dotenv()

logging.basicConfig(
    filename="monitoring/api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================================================
#  GLOBAL CONFIG
# ============================================================
API_KEY = os.getenv("API_KEY", "defaultapikey")
ADMIN_KEY = os.getenv("ADMIN_API_KEY", API_KEY)       # allow admin key
USER_KEY = os.getenv("USER_API_KEY", API_KEY)         # allow user key

API_KEY_NAME = "x-api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

RATE_LIMIT = 60       # allowed requests per window
WINDOW_SIZE = 60      # seconds
rate_limit_store = {} # per-key request counters

MODEL_DIR = "models"

START_TIME = time.time()
TOTAL_REQUESTS = 0
PREDICTION_COUNT = 0
LAST_ERROR = None
LAST_MODEL_LOAD_TIME = None
LAST_DRIFT_CHECK = None
LAST_DRIFT_RESULT = None

explainer = None
model = None


# ============================================================
#  RATE LIMITER
# ============================================================
def check_rate_limit(api_key: str):
    now = time.time()

    if api_key not in rate_limit_store:
        rate_limit_store[api_key] = []

    # Keep timestamps within last window
    request_times = [t for t in rate_limit_store[api_key] if now - t < WINDOW_SIZE]
    rate_limit_store[api_key] = request_times

    if len(request_times) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    rate_limit_store[api_key].append(now)


# ============================================================
#  API KEY VERIFICATION + ROLE CHECK
# ============================================================
def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key in [API_KEY, ADMIN_KEY, USER_KEY]:
        check_rate_limit(api_key)
        return api_key

    raise HTTPException(status_code=401, detail="Invalid or missing API key")


def admin_only(api_key: str = Security(api_key_header)):
    if api_key == ADMIN_KEY:
        return True
    raise HTTPException(status_code=403, detail="Admin permissions required")


# ============================================================
#  MODEL LOADING
# ============================================================
def load_latest_model():
    global LAST_MODEL_LOAD_TIME

    pattern = os.path.join(MODEL_DIR, "churn_pipeline_v*.joblib")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError("No versioned models found in models/ folder.")

    versions = []
    for f in files:
        match = re.search(r"v(\d+)", os.path.basename(f))
        if match:
            versions.append((f, int(match.group(1))))

    if not versions:
        raise FileNotFoundError("No properly versioned model files like churn_pipeline_v1.joblib")

    latest_file = max(versions, key=lambda x: x[1])[0]

    LAST_MODEL_LOAD_TIME = datetime.utcnow()

    print(f"Loaded model: {os.path.basename(latest_file)}")
    return joblib.load(latest_file)



def load_shap_explainer():
    global explainer

    try:
        # Get model from pipeline
        raw_model = model.named_steps["model"]
        explainer = shap.TreeExplainer(raw_model)
        print("SHAP explainer initialized")
    except Exception as e:
        explainer = None
        print("SHAP initialization error:", e)



# ============================================================
#  STARTUP INITIALIZE
# ============================================================

model = load_latest_model()
load_shap_explainer()

app = FastAPI(title="Churn Prediction API")

import time
import psycopg2
from psycopg2 import OperationalError

def wait_for_db():
    import psycopg2, time
    print("⏳ Waiting for PostgreSQL...")

    while True:
        try:
            conn = psycopg2.connect(
                host="churn_db",
                port=5432,
                user="postgres",
                password="postgres",
                database="churn_db"
            )
            conn.close()
            print("✅ PostgreSQL is ready!")
            break

        except Exception:
            print("❌ DB not ready, retrying in 1 sec...")
            time.sleep(1)




@app.on_event("startup")
def startup_event():
    wait_for_db()
    init_db()  # create your tables AFTER DB is ready
    print("🚀 Startup completed!")




# ============================================================
#  MIDDLEWARE — Count API Requests + Latency
# ============================================================
@app.middleware("http")
async def count_requests(request: Request, call_next):
    global TOTAL_REQUESTS, LAST_ERROR
    TOTAL_REQUESTS += 1

    start = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        LAST_ERROR = str(e)
        raise e

    latency = time.time() - start
    response.headers["X-API-Latency"] = f"{latency:.4f}s"

    return response


# ============================================================
#  INPUT SCHEMA
# ============================================================
class Customer(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float
    contract_type: str
    gender: str
    senior_citizen: int


# ============================================================
#  ROOT ENDPOINT
# ============================================================
@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}


# ============================================================
#  PREDICT — SINGLE
# ============================================================
@app.post("/predict")
def predict_churn(
    customer: Customer,
    background_tasks: BackgroundTasks,
    authorized: str = Depends(verify_api_key)
):
    global PREDICTION_COUNT

    df = pd.DataFrame([customer.dict()])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    PREDICTION_COUNT += 1

    if PREDICTION_COUNT % 30 == 0:
        background_tasks.add_task(background_drift_scan)

    background_tasks.add_task(log_prediction, {
        "input": customer.dict(),
        "prediction": int(prediction),
        "probability": float(probability)
    })

    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability)
    }


# ============================================================
#  PREDICT — BATCH JSON
# ============================================================
@app.post("/predict-batch-json")
def predict_batch_json(
    customers: list[Customer],
    background_tasks: BackgroundTasks,
    authorized: str = Depends(verify_api_key)
):
    global PREDICTION_COUNT

    df = pd.DataFrame([c.dict() for c in customers])
    preds = model.predict(df).tolist()
    probs = model.predict_proba(df)[:, 1].tolist()

    PREDICTION_COUNT += len(preds)

    results = []
    for inp, pred, prob in zip(customers, preds, probs):
        background_tasks.add_task(log_prediction, {
            "input": inp.dict(),
            "prediction": int(pred),
            "probability": float(prob)
        })
        results.append({
            "input": inp.dict(),
            "churn_prediction": int(pred),
            "churn_probability": float(prob)
        })

    return {"results": results}


# ============================================================
#  PREDICT — CSV
# ============================================================
@app.post("/predict-batch-csv")
async def predict_batch_csv(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    authorized: str = Depends(verify_api_key)
):
    global PREDICTION_COUNT

    contents = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        return {"error": f"CSV read failed: {str(e)}"}

    required_cols = ["tenure", "monthly_charges", "total_charges", "contract_type", "gender", "senior_citizen"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        return {"error": "Missing required columns", "missing": missing}

    preds = model.predict(df).tolist()
    probs = model.predict_proba(df)[:, 1].tolist()
    
    # Trigger drift scan after batch
    background_tasks.add_task(background_drift_scan)


    PREDICTION_COUNT += len(preds)

    df["churn_prediction"] = preds
    df["churn_probability"] = probs

    if background_tasks:
        for _, row in df.iterrows():
            background_tasks.add_task(log_prediction, {
                "input": row.to_dict(),
                "prediction": int(row["churn_prediction"]),
                "probability": float(row["churn_probability"])
            })

    return {
        "filename": file.filename,
        "rows": len(df),
        "preview": df.head(10).to_dict(orient="records")
    }


@app.post("/explain")
def explain(customer: Customer, authorized: bool = Depends(verify_api_key)):

    if explainer is None:
        return {"error": "SHAP explainer not available"}

    try:
        df = pd.DataFrame([customer.dict()])

        preprocessor = model.named_steps["preprocessor"]
        raw_model = model.named_steps["model"]

        X_transformed = preprocessor.transform(df)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        shap_values_all = explainer.shap_values(X_transformed)

        # Make SHAP safe for binary-class + RF
        if isinstance(shap_values_all, list):
            shap_values = shap_values_all[-1]  # pick last class (1)
        else:
            shap_values = shap_values_all

        feature_names = preprocessor.get_feature_names_out()

        # Flatten shap_values and convert to list
        import numpy as np
        if isinstance(shap_values, np.ndarray):
            shap_values_flat = shap_values.flatten()
        else:
            shap_values_flat = np.array(shap_values).flatten()
        
        # Convert to Python list of floats
        shap_values_list = [float(v) for v in shap_values_flat]
        
        # Handle expected_value (might be array or scalar)
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            # For binary classification, take the positive class
            base_value = float(expected_value[-1] if isinstance(expected_value, list) else expected_value.flatten()[-1])
        else:
            base_value = float(expected_value)

        return {
            "features": feature_names.tolist(),
            "shap_values": shap_values_list,
            "base_value": base_value,
        }
    
    except Exception as e:
        import traceback
        return {
            "error": f"Explanation failed: {str(e)}",
            "traceback": traceback.format_exc()
        }
        
        
@app.post("/explain-simple")
def explain_simple(customer: Customer, authorized: bool = Depends(verify_api_key)):

    if explainer is None:
        return {"error": "SHAP explainer not available"}

    try:
        df = pd.DataFrame([customer.dict()])
        preprocessor = model.named_steps["preprocessor"]

        X_transformed = preprocessor.transform(df)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        shap_values_all = explainer.shap_values(X_transformed)

        if isinstance(shap_values_all, list):
            shap_values = shap_values_all[-1]
        else:
            shap_values = shap_values_all

        feature_names = preprocessor.get_feature_names_out()

        # Flatten shap_values to ensure it's 1D
        import numpy as np
        if isinstance(shap_values, np.ndarray):
            shap_values_flat = shap_values.flatten()
        else:
            shap_values_flat = np.array(shap_values).flatten()
        
        # Create pairs with explicit conversion
        shap_pairs = []
        for i, feature in enumerate(feature_names):
            if i < len(shap_values_flat):
                shap_pairs.append((str(feature), float(shap_values_flat[i])))
        
        # Sort by absolute value
        shap_sorted = sorted(shap_pairs, key=lambda x: abs(x[1]), reverse=True)

        return {
            "top_factors": [
                {"feature": f, "impact": v}
                for f, v in shap_sorted[:5]
            ]
        }
    
    except Exception as e:
        import traceback
        return {
            "error": f"Explanation failed: {str(e)}",
            "traceback": traceback.format_exc()
        }

# ============================================================
#  DRIFT + RETRAIN
# ============================================================
def background_drift_scan():
    global LAST_DRIFT_CHECK, LAST_DRIFT_RESULT

    result = check_drift()
    LAST_DRIFT_CHECK = datetime.utcnow()
    LAST_DRIFT_RESULT = result

    if result.get("has_drift"):
        print("⚠️ Drift detected → initiating retrain...")
        background_retrain()


def background_retrain():
    global model
    try:
        retrain_model()
        model = load_latest_model()
        load_shap_explainer()
        print("Model retrained & reloaded successfully")
    except Exception as e:
        print("Retrain failed:", e)


@app.post("/retrain")
def retrain_endpoint(
    background_tasks: BackgroundTasks,
    auth: bool = Depends(admin_only)
):
    background_tasks.add_task(background_retrain)
    return {"message": "Retrain started in background"}


# ============================================================
#  DIAGNOSTIC ENDPOINTS
# ============================================================
@app.get("/drift-status")
def drift_status(auth: str = Depends(verify_api_key)):
    return {
        "last_checked": str(LAST_DRIFT_CHECK),
        "drift_result": LAST_DRIFT_RESULT
    }


@app.get("/backend-stats")
def backend_stats(auth: str = Depends(verify_api_key)):
    return {
        "total_requests": TOTAL_REQUESTS,
        "prediction_count": PREDICTION_COUNT,
        "last_error": LAST_ERROR,
        "server_start_time": datetime.utcfromtimestamp(START_TIME).isoformat()
    }
    

# Add these endpoints to your api.py file

@app.get("/models/list")
def list_models(auth: str = Depends(verify_api_key)):
    """List all available model versions"""
    pattern = os.path.join(MODEL_DIR, "churn_pipeline_v*.joblib")
    files = glob.glob(pattern)
    
    models = []
    for f in files:
        match = re.search(r"v(\d+)", os.path.basename(f))
        if match:
            version = int(match.group(1))
            file_size = os.path.getsize(f)
            modified_time = datetime.fromtimestamp(os.path.getmtime(f))
            
            models.append({
                "version": version,
                "filename": os.path.basename(f),
                "size_mb": round(file_size / (1024 * 1024), 2),
                "modified": modified_time.isoformat()
            })
    
    # Sort by version descending
    models.sort(key=lambda x: x["version"], reverse=True)
    
    return {
        "models": models,
        "total_count": len(models),
        "currently_loaded": LAST_MODEL_LOAD_TIME.isoformat() if LAST_MODEL_LOAD_TIME else None
    }


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round(time.time() - start, 4)

    logging.info(
        f"{request.method} {request.url.path} "
        f"Status {response.status_code} "
        f"Time {duration}s"
    )

    return response

@app.post("/models/load")
def load_model_version(
    version: int,
    background_tasks: BackgroundTasks,
    auth: bool = Depends(admin_only)
):
    """Load a specific model version (Admin only)"""
    global model, LAST_MODEL_LOAD_TIME
    
    model_path = os.path.join(MODEL_DIR, f"churn_pipeline_v{version}.joblib")
    
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Model version {version} not found"
        )
    
    try:
        model = joblib.load(model_path)
        LAST_MODEL_LOAD_TIME = datetime.utcnow()
        
        # Reload SHAP explainer for new model
        background_tasks.add_task(load_shap_explainer)
        
        return {
            "message": f"Model version {version} loaded successfully",
            "model_path": model_path,
            "loaded_at": LAST_MODEL_LOAD_TIME.isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/uptime")
def get_uptime():
    uptime_seconds = int(time.time() - START_TIME)
    return {
        "uptime_seconds": uptime_seconds,
        "uptime": f"{uptime_seconds//3600:02d}:{(uptime_seconds//60)%60:02d}:{uptime_seconds%60:02d}"
    }



@app.post("/explain-batch")
def explain_batch(customers: list[Customer], authorized: bool = Depends(verify_api_key)):

    if explainer is None:
        return {"error": "SHAP explainer not available"}

    df = pd.DataFrame([c.dict() for c in customers])
    preprocessor = model.named_steps["preprocessor"]

    X_transformed = preprocessor.transform(df)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    shap_values = explainer.shap_values(X_transformed)[1]
    feature_names = preprocessor.get_feature_names_out()

    # Mean absolute SHAP values (global importance)
    importance = dict(zip(feature_names, abs(shap_values).mean(axis=0)))

    sorted_imp = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    return {
        "feature_importance": sorted_imp,
        "shap_values": shap_values.tolist(),
        "feature_names": feature_names.tolist()
    }



@app.post("/batch-report")
def batch_report(customers: list[Customer], authorized: bool = Depends(verify_api_key)):

    df = pd.DataFrame([c.dict() for c in customers])
    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    df["prediction"] = preds
    df["probability"] = probs

    report_path = "monitoring/batch_report.pdf"

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(report_path)
    story = []

    story.append(Paragraph("Churn Model Batch Report", styles["Title"]))
    story.append(Paragraph(f"Total Records: {len(df)}", styles["Normal"]))

    # Summary stats
    story.append(Paragraph("Summary Statistics:", styles["Heading2"]))
    story.append(Paragraph(df.describe().to_html(), styles["Normal"]))

    doc.build(story)

    return FileResponse(report_path, filename="batch_report.pdf")