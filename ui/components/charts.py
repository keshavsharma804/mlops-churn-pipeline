import streamlit as st
import pandas as pd
import plotly.express as px
import json

def load_logs(path="monitoring/prediction_logs.jsonl"):
    rows = []
    try:
        with open(path, "r") as f:
            for l in f:
                obj = json.loads(l)
                obj["input"].update({
                    "prediction": obj["prediction"],
                    "probability": obj["probability"],
                    "timestamp": obj["timestamp"]
                })
                rows.append(obj["input"])
    except:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def show_monitoring_charts():
    st.header("📡 Real-Time Monitoring")

    df = load_logs()

    if df.empty:
        st.warning("No logs found yet.")
        return

    # Prediction probability distribution
    fig = px.histogram(df, x="probability", nbins=20, title="Prediction Probability Distribution")
    st.plotly_chart(fig, use_container_width=True)

    # Predictions over time
    df_sorted = df.sort_values("timestamp")
    fig = px.line(df_sorted, x="timestamp", y="prediction", title="Prediction Trend Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # Feature drift histogram
    if "tenure" in df.columns:
        fig = px.histogram(df, x="tenure", title="Feature Drift: Tenure Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional: Monthly charges drift
    if "monthly_charges" in df.columns:
        fig = px.histogram(df, x="monthly_charges", title="Feature Drift: Monthly Charges")
        st.plotly_chart(fig, use_container_width=True)

    # Additional: Churn probability trend
    fig = px.line(df_sorted, x="timestamp", y="probability", title="Churn Probability Trend")
    st.plotly_chart(fig, use_container_width=True)

