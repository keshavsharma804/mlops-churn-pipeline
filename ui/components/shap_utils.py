import shap
import matplotlib.pyplot as plt
import streamlit as st

def plot_shap_force(explainer, values, features):
    st.subheader("SHAP Force Plot")
    fig = shap.force_plot(explainer.expected_value[1], values, features, matplotlib=True)
    st.pyplot(fig)

def plot_shap_waterfall(explainer, shap_values):
    st.subheader("Waterfall Plot")
    fig = shap.plots.waterfall(shap_values)
    st.pyplot(fig)

def plot_shap_summary(shap_values, feature_names):
    st.subheader("Feature Impact Summary")
    fig = shap.summary_plot(shap_values, feature_names, show=False)
    st.pyplot(fig)
