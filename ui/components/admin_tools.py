import streamlit as st
import requests
import pandas as pd

def admin_panel(API_URL, headers):
    st.title("🛠 Admin Panel")

    st.subheader("📦 Model Versions")
    r = requests.get(f"{API_URL}/model-list", headers=headers).json()
    df = pd.DataFrame(r["models"])
    st.table(df)

    st.subheader("🔄 Load Specific Model")
    version = st.number_input("Enter Model Version", min_value=1, step=1)
    if st.button("Load Model"):
        res = requests.post(f"{API_URL}/models/load?version={version}", headers=headers).json()
        st.write(res)

    st.subheader("♻ Retrain Model (Background)")
    if st.button("Trigger Retrain"):
        res = requests.post(f"{API_URL}/retrain", headers=headers).json()
        st.success(res)
