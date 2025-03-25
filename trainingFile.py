import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple


def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads and preprocesses the dataset.
    
    Args:
        file_path (str): Path to the dataset CSV file.
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Processed features (X) and target variable (y).
    """
    df = pd.read_csv(file_path)
    df_filtered = df[df["vomitoxin_ppb"] <= 20000].copy()
    df_filtered["vomitoxin_ppb"] = np.log1p(df_filtered["vomitoxin_ppb"])
    
    # Extract spectral data (excluding 'hsi_id' and 'vomitoxin_ppb')
    spectral_data = df_filtered.iloc[:, 1:-1]
    
    # Feature Engineering
    df_filtered["mean_reflectance"] = spectral_data.mean(axis=1)
    df_filtered["std_reflectance"] = spectral_data.std(axis=1)
    df_filtered["first_order_derivative"] = spectral_data.diff(axis=1).mean(axis=1)
    df_filtered["NDSI_50_150"] = (spectral_data.iloc[:, 50] - spectral_data.iloc[:, 150]) / (
        spectral_data.iloc[:, 50] + spectral_data.iloc[:, 150] + 1e-6
    )
    
    X = df_filtered.iloc[:, 1:-1]  # Features (excluding 'hsi_id' and target)
    y = df_filtered["vomitoxin_ppb"]  # Target variable
    
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Splits the dataset into training and testing sets.
    
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        test_size (float, optional): Proportion of test data. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
    
    Returns:
        Tuple: Split datasets (X_train, X_test, y_train, y_test).
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_dnn_model(input_shape: int) -> keras.Model:
    """
    Builds a deep neural network (DNN) model for regression.
    
    Args:
        input_shape (int): Number of input features.
    
    Returns:
        keras.Model: Compiled Keras model.
    """
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(input_shape,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)  # Output layer for regression
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_model(model: keras.Model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                epochs: int = 100, batch_size: int = 16) -> keras.callbacks.History:
    """
    Trains the deep learning model.
    
    Args:
        model (keras.Model): Compiled model.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target values.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test target values.
        epochs (int, optional): Number of training epochs. Defaults to 100.
        batch_size (int, optional): Batch size. Defaults to 16.
    
    Returns:
        keras.callbacks.History: Training history object.
    """
    return model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)


def evaluate_model(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Evaluates the model and prints performance metrics.
    
    Args:
        model (keras.Model): Trained model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): True target values.
    """
    y_pred = model.predict(X_test).flatten()
    y_pred_original = np.expm1(y_pred)
    y_test_original = np.expm1(y_test)
    
    mae = mean_absolute_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    r2 = r2_score(y_test_original, y_pred_original)
    
    print(f"📌 DNN Model Performance:")
    print(f"✅ MAE: {mae:.2f} ppb")
    print(f"✅ RMSE: {rmse:.2f} ppb")
    print(f"✅ R² Score: {r2:.4f}")


# Main execution
if __name__ == "__main__":
    FILE_PATH = "MLE-Assignment.csv"  # Replace with actual path
    X, y = load_and_preprocess_data(FILE_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = build_dnn_model(X_train.shape[1])
    train_model(model, X_train, y_train, X_test, y_test)
    evaluate_model(model, X_test, y_test)
