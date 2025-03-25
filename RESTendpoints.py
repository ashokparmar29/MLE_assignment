from fastapi import FastAPI, File, UploadFile
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import io

app = FastAPI()

# Load the trained model and scaler
def load_model():
    model = tf.keras.models.load_model("dnn_model.h5", compile=False)

    # Recompile the model with correct loss & metrics
    model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError()])

    return model

model = load_model()  # Load model once at startup

# Preprocess the data
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df_filtered = df.copy()
    df_filtered["mean_reflectance"] = df.mean(axis=1)
    df_filtered["std_reflectance"] = df.std(axis=1)
    df_filtered["first_order_derivative"] = df.diff(axis=1).mean(axis=1)
    df_filtered["NDSI_50_150"] = (df.iloc[:, 50] - df.iloc[:, 150]) / (df.iloc[:, 50] + df.iloc[:, 150] + 1e-6)
    
    return df_filtered

# Predict function
def predict(model, X: pd.DataFrame) -> np.ndarray:
    y_pred = model.predict(X).flatten()
    y_pred_original = np.expm1(y_pred)  # Inverse log transformation
    return y_pred_original

# Endpoint to upload file and make predictions
@app.post("/predict/")
async def upload_file(file: UploadFile = File(...)):
    # Read CSV file
    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))

    # Preprocess data
    X_processed = preprocess_data(df)

    # Make predictions
    predictions = predict(model, X_processed)

    # Return results as JSON
    df_results = df.copy()
    df_results["Predicted_Vomitoxin_ppb"] = predictions

    return df_results.to_dict(orient="records")
