import streamlit as st
import requests
import pandas as pd
from components.charts import show_monitoring_charts
from components.shap_utils import *
from components.styling import apply_theme
from components.admin_tools import admin_panel

# ----------------------------
# CONFIG
# ----------------------------
API_URL = "http://churn_backend:8000"
USER_KEY = "user1234key"
ADMIN_KEY = "admin9999key"

headers = {"x-api-key": USER_KEY}

apply_theme()

st.sidebar.title("🚀 Churn AI Platform")

menu = st.sidebar.radio("Navigation", [
    "🏠 Dashboard",
    "📍 Single Prediction",
    "📂 Batch Prediction",
    "📊 Explainability",
    "📡 Monitoring",
    "🛠 Admin"
])

# =====================================================
# 1. DASHBOARD
# =====================================================
if menu == "🏠 Dashboard":
    st.title("🏠 Dashboard")

    try:
        health = requests.get(f"{API_URL}/health").json()
        uptime = requests.get(f"{API_URL}/uptime").json()
        stats = requests.get(f"{API_URL}/backend-stats", headers=headers).json()

        col1, col2, col3 = st.columns(3)
        col1.metric("Status", health["status"])
        col2.metric("Uptime", uptime["uptime"])
        col3.metric("Total Requests", stats["total_requests"])

    except Exception as e:
        st.error(f"❌ Cannot connect to backend: {e}")


# =====================================================
# 2. SINGLE PREDICTION
# =====================================================
elif menu == "📍 Single Prediction":
    st.title("📍 Predict Single Customer")

    col1, col2 = st.columns(2)
    tenure = col1.number_input("Tenure", 0, 100, 10)
    senior = col2.number_input("Senior Citizen", 0, 1, 0)
    monthly = col1.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total = col2.number_input("Total Charges", 0.0, 10000.0, 500.0)
    contract = col1.selectbox("Contract", ["month-to-month", "one-year", "two-year"])
    gender = col2.selectbox("Gender", ["male", "female"])

    payload = {
        "tenure": tenure,
        "monthly_charges": monthly,
        "total_charges": total,
        "contract_type": contract,
        "gender": gender,
        "senior_citizen": senior
    }

    if st.button("🔮 Predict Churn"):
        with st.spinner("⏳ Running prediction..."):
            try:
                res = requests.post(f"{API_URL}/predict", json=payload, headers=headers)
                st.write("Status:", res.status_code)
                st.write("Response:", res.text)

                if res.status_code == 200:
                    r = res.json()
                    st.success(f"Prediction: {'Churn' if r['churn_prediction'] else 'No Churn'}")
                    st.metric("Churn Probability", f"{r['churn_probability']:.4f}")

                else:
                    st.error("Prediction failed")

            except Exception as e:
                st.error(f"❌ Error: {e}")


# =====================================================
# 3. BATCH PREDICTION
# =====================================================
elif menu == "📂 Batch Prediction":
    st.title("📂 Batch Prediction")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.subheader("📄 Input Preview")
        st.dataframe(df.head())

        files = {"file": (file.name, file.getvalue(), "text/csv")}

        if st.button("⚙️ Run Batch Prediction"):
            with st.spinner("Processing Batch..."):
                try:
                    r = requests.post(f"{API_URL}/predict-batch-csv", files=files, headers=headers)

                    st.write("Status:", r.status_code)
                    st.write("Response:", r.text)

                    if r.status_code == 200:
                        st.success("Batch prediction completed!")
                        st.json(r.json())
                    else:
                        st.error("Batch processing failed")

                except Exception as e:
                    st.error(f"❌ Error: {e}")


# =====================================================
# 4. SHAP EXPLAINABILITY
# =====================================================
elif menu == "📊 Explainability":
    st.title("📊 Model Explainability")

    st.info("Run a prediction first on the Single Prediction page.")

    # Optional: In future we can add a SHAP summary plot handled by components/shap_utils.py.


# =====================================================
# 5. MONITORING
# =====================================================
elif menu == "📡 Monitoring":
    st.title("📡 Monitoring & Metrics")
    show_monitoring_charts()


# =====================================================
# 6. ADMIN TOOLS
# =====================================================
elif menu == "🛠 Admin":
    st.title("🛠 Admin Control Panel")
    headers = {"x-api-key": ADMIN_KEY}
    admin_panel(API_URL, headers)
