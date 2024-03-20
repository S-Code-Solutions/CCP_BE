import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(file_path):
    """
    Load data from a file path.

    :param file_path: String, path to the file containing the dataset.
    :return: DataFrame, loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return data
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None

def clean_data(data):
    """
    Perform data cleaning operations such as filling missing values, removing duplicates, etc.

    :param data: DataFrame, raw data.
    :return: DataFrame, cleaned data.
    """
    # Example: fill missing values with the median
    data_filled = data.fillna(data.median())
    # Example: remove duplicates
    data_cleaned = data_filled.drop_duplicates()
    print("Data cleaned.")
    return data_cleaned

def feature_engineering(data):
    """
    Create new features from existing ones or perform transformations.

    :param data: DataFrame, cleaned data.
    :return: DataFrame, data with new features.
    """
    # Example: Add a new feature that is a combination of two existing features
    # This is just a placeholder example. Replace with actual feature engineering steps.
    if 'feature1' in data.columns and 'feature2' in data.columns:
        data['new_feature'] = data['feature1'] * data['feature2']
    print("Feature engineering done.")
    return data

def scale_features(data, scaler_path=None):
    """
    Scale the feature values using StandardScaler or a previously fitted scaler.

    :param data: DataFrame, data to scale.
    :param scaler_path: Optional; string, path to a saved scaler object. If None, fit a new scaler.
    :return: DataFrame, scaled data.
    """
    features = data.select_dtypes(include=[float, int]).columns
    if scaler_path:
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()
        scaler.fit(data[features])

    data[features] = scaler.transform(data[features])
    print("Features scaled.")

    # Optionally save the scaler for future use if it was newly fitted
    if not scaler_path:
        joblib.dump(scaler, 'models/new_scaler.pkl')
    return data

# Example usage:
if __name__ == '__main__':
    file_path = 'path/to/your/data.csv'
    data = load_data(file_path)
    if data is not None:
        data_cleaned = clean_data(data)
        data_fe = feature_engineering(data_cleaned)
        data_scaled = scale_features(data_fe, scaler_path='path/to/your/scaler.pkl')
        # Now data_scaled is ready for use with machine learning models
