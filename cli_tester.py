import requests
import json
import time
import pandas as pd
from io import BytesIO

BASE_URL = "http://127.0.0.1:8000"

API_KEY = "user1234key"
ADMIN_KEY = "admin9999key"
 
# ======================================

HEADERS = {"x-api-key": API_KEY}
ADMIN_HEADERS = {"x-api-key": ADMIN_KEY}


def print_section(title):
    print("\n" + "="*60)
    print(title)
    print("="*60)


# -----------------------------------------
# 1. HEALTH CHECK
# -----------------------------------------
def test_health():
    print_section("Health Check")
    r = requests.get(f"{BASE_URL}/health")
    print(r.status_code, r.json())


# -----------------------------------------
# 2. UPTIME
# -----------------------------------------
def test_uptime():
    print_section("Uptime")
    r = requests.get(f"{BASE_URL}/uptime")
    print(r.status_code, r.json())


# -----------------------------------------
# 3. SINGLE PREDICTION
# -----------------------------------------
def test_single_predict():
    print_section("Single Prediction")

    payload = {
        "tenure": 5,
        "monthly_charges": 82.4,
        "total_charges": 412.0,
        "contract_type": "month-to-month",
        "gender": "male",
        "senior_citizen": 0
    }

    r = requests.post(f"{BASE_URL}/predict", json=payload, headers=HEADERS)
    print(r.status_code, r.json())


# -----------------------------------------
# 4. BATCH JSON PREDICTION
# -----------------------------------------
def test_batch_json():
    print_section("Batch JSON Prediction")

    payload = [
        {
            "tenure": 10,
            "monthly_charges": 50,
            "total_charges": 500,
            "contract_type": "one-year",
            "gender": "female",
            "senior_citizen": 0
        },
        {
            "tenure": 2,
            "monthly_charges": 80,
            "total_charges": 160,
            "contract_type": "month-to-month",
            "gender": "male",
            "senior_citizen": 1
        }
    ]

    r = requests.post(f"{BASE_URL}/predict-batch-json", json=payload, headers=HEADERS)
    print(r.status_code, r.json())


# -----------------------------------------
# 5. BATCH CSV PREDICTION
# -----------------------------------------
def test_batch_csv():
    print_section("Batch CSV Prediction")

    df = pd.DataFrame([
        [1, 70, 70, "month-to-month", "male", 0],
        [20, 40, 800, "one-year", "female", 0]
    ], columns=["tenure", "monthly_charges", "total_charges", "contract_type", "gender", "senior_citizen"])

    csv_bytes = df.to_csv(index=False).encode()

    files = {"file": ("sample.csv", BytesIO(csv_bytes), "text/csv")}

    r = requests.post(f"{BASE_URL}/predict-batch-csv", files=files, headers=HEADERS)
    print(r.status_code, r.json())


# -----------------------------------------
# 6. EXPLAINABILITY (SHAP)
# -----------------------------------------
def test_explain():
    print_section("Explainability (SHAP Full)")

    payload = {
        "tenure": 5,
        "monthly_charges": 70,
        "total_charges": 350,
        "contract_type": "one-year",
        "gender": "female",
        "senior_citizen": 0
    }

    r = requests.post(f"{BASE_URL}/explain", json=payload, headers=HEADERS)
    result = r.json()
    print(f"Status: {r.status_code}")
    print(f"Features: {len(result.get('features', []))} total")
    print(f"Base Value: {result.get('base_value')}")
    print(f"Top 3 SHAP values: {result.get('shap_values', [])[:3]}")


def test_explain_simple():
    print_section("Explainability (SHAP Simple)")

    payload = {
        "tenure": 5,
        "monthly_charges": 70,
        "total_charges": 350,
        "contract_type": "one-year",
        "gender": "female",
        "senior_citizen": 0
    }

    r = requests.post(f"{BASE_URL}/explain-simple", json=payload, headers=HEADERS)
    print(r.status_code, r.json())


# -----------------------------------------
# 7. MODEL LIST + LOAD
# -----------------------------------------
def test_model_list():
    print_section("Model List")
    r = requests.get(f"{BASE_URL}/models/list", headers=HEADERS)  # FIX: Added headers
    print(r.status_code, r.json())


def test_model_load():
    print_section("Model Load (Admin Only)")
    r = requests.post(f"{BASE_URL}/models/load?version=1", headers=ADMIN_HEADERS)
    print(r.status_code, r.json())


# -----------------------------------------
# 8. RETRAIN MODEL
# -----------------------------------------
def test_retrain():
    print_section("Retrain Model (Admin Only)")
    r = requests.post(f"{BASE_URL}/retrain", headers=ADMIN_HEADERS)
    print(r.status_code, r.json())


# -----------------------------------------
# 9. BACKEND STATS
# -----------------------------------------
def test_backend_stats():
    print_section("Backend Stats")
    r = requests.get(f"{BASE_URL}/backend-stats", headers=HEADERS)
    print(r.status_code, r.json())


# -----------------------------------------
# 10. DRIFT STATUS
# -----------------------------------------
def test_drift_status():
    print_section("Drift Status")
    r = requests.get(f"{BASE_URL}/drift-status", headers=HEADERS)
    print(r.status_code, r.json())


# -----------------------------------------
# 11. RATE LIMITING TEST
# -----------------------------------------
def test_rate_limit():
    print_section("Rate Limiting Test (sending 65 quick requests)")

    success_count = 0
    rate_limited_count = 0
    
    for i in range(65):
        r = requests.get(f"{BASE_URL}/health", headers=HEADERS)
        if r.status_code == 200:
            success_count += 1
        elif r.status_code == 429:
            rate_limited_count += 1
            print(f"Request {i+1}: Rate limited!")
        time.sleep(0.01)
    
    print(f"\nResults: {success_count} successful, {rate_limited_count} rate-limited")


# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == "__main__":

    print_section("STARTING API TESTING SUITE")

    test_health()
    test_uptime()

    test_single_predict()
    test_batch_json()
    test_batch_csv()

    test_explain()
    test_explain_simple()

    test_model_list()
    test_model_load()

    test_backend_stats()
    test_drift_status()

    # Uncomment if you want to test retraining:
    # test_retrain()

    # Uncomment if you want to test rate limiting:
    # test_rate_limit()

    print_section("ALL TESTS COMPLETED ✅")