import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_model(model_path):
    """
    Loads a machine learning model from the specified path.

    :param model_path: Path to the .pkl file where the model is saved.
    :return: Loaded model.
    """
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def preprocess_data(data, scaler_path=None):
    """
    Preprocesses the data for prediction. This can include scaling, normalization,
    and other necessary steps before feeding data into the model.

    :param data: The input data to preprocess.
    :param scaler_path: Optional path to a scaler object for data normalization.
    :return: Preprocessed data.
    """
    if scaler_path:
        try:
            scaler = joblib.load(scaler_path)
            preprocessed_data = scaler.transform(data)
            print("Data scaled successfully.")
        except Exception as e:
            print(f"Error scaling data: {e}")
            preprocessed_data = data  # Return original data if scaling fails
    else:
        # If no scaler is provided, return the data as is or implement other preprocessing steps
        preprocessed_data = data
    
    return preprocessed_data

def create_time_features(data):
    """
    Optional: Creates time-based features from the datetime index or columns in the dataset.
    Useful for time series models that may benefit from information about the hour, day, month, etc.

    :param data: The input DataFrame with datetime index or columns.
    :return: DataFrame with additional time-based features.
    """
    if isinstance(data, pd.DataFrame) and not data.index.empty:
        data['hour'] = data.index.hour
        data['dayofweek'] = data.index.dayofweek
        data['month'] = data.index.month
        print("Time features created successfully.")
    else:
        print("Data does not have a datetime index. No time features added.")
    return data

# Example usage
if __name__ == "__main__":
    # Example demonstrating how to use utility functions
    # Replace with actual data paths and preprocessing logic as necessary
    model_path = 'models/my_model.pkl'
    scaler_path = 'models/my_scaler.pkl'
    model = load_model(model_path)
    
    # Dummy input data for demonstration; replace with actual data
    dummy_data = np.array([[1, 2, 3], [4, 5, 6]])
    preprocessed_data = preprocess_data(dummy_data, scaler_path=scaler_path)
    
    # Assuming the model and preprocessed data are ready
    if model is not None:
        predictions = model.predict(preprocessed_data)
        print(predictions)
