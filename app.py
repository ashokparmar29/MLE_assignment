import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model and scaler
def load_model():
    model = tf.keras.models.load_model("dnn_model.h5", compile=False)  # Ensure model is in the working directory

    # Recompile the model with correct loss & metrics
    model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError()])

    return model

# Preprocess uploaded data
def preprocess_data(df):
    df_filtered = df.copy()
    df_filtered["mean_reflectance"] = df.mean(axis=1)
    df_filtered["std_reflectance"] = df.std(axis=1)
    df_filtered["first_order_derivative"] = df.diff(axis=1).mean(axis=1)
    df_filtered["NDSI_50_150"] = (df.iloc[:, 50] - df.iloc[:, 150]) / (df.iloc[:, 50] + df.iloc[:, 150] + 1e-6)
    
    return df_filtered

# Make predictions
def predict(model, X):
    y_pred = model.predict(X).flatten()
    y_pred_original = np.expm1(y_pred)  # Inverse log transformation
    return y_pred_original

# Streamlit UI
st.title("🌾 Spectral Data DON Prediction App")
st.write("Upload your spectral data and get real-time vomitoxin (DON) predictions.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview:")
    st.dataframe(df.head())
    
    model = load_model()
    X_processed = preprocess_data(df)
    predictions = predict(model, X_processed)
    
    st.write("### Predicted Vomitoxin Levels:")
    df_results = df.copy()
    df_results["Predicted_Vomitoxin_ppb"] = predictions
    st.dataframe(df_results)
    
    st.success("✅ Prediction Completed!")
