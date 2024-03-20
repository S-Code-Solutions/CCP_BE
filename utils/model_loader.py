import joblib

def load_arima_model(model_path):
    """
    Load the ARIMA model from the specified .pkl file.

    :param model_path: string, path to the saved ARIMA model (.pkl file).
    :return: Loaded ARIMA model.
    """
    try:
        arima_model = joblib.load(model_path)
        print("ARIMA model loaded successfully.")
        return arima_model
    except Exception as e:
        print(f"Error loading ARIMA model: {e}")
        return None

def load_lstm_model(model_path):
    """
    Load the LSTM model from the specified .pkl file.

    :param model_path: string, path to the saved LSTM model (.pkl file).
    :return: Loaded LSTM model.
    """
    try:
        lstm_model = joblib.load(model_path)
        print("LSTM model loaded successfully.")
        return lstm_model
    except Exception as e:
        print(f"Error loading LSTM model: {e}")
        return None

# Example usage
if __name__ == "__main__":
    arima_model_path = "path/to/your/arima_model.pkl"
    lstm_model_path = "path/to/your/lstm_model.pkl"

    # Load models
    arima_model = load_arima_model(arima_model_path)
    lstm_model = load_lstm_model(lstm_model_path)

    # You can now use these models for prediction in your application
