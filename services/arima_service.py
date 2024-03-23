import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMAResults
from sklearn.metrics import mean_squared_error

class ARIMAService:
    def __init__(self, model_path='model/arima_model.pkl'):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        """
        Loads the ARIMA model from the specified file.
        """
        try:
            loaded_model = ARIMAResults.load(self.model_path)
            print("ARIMA model loaded successfully.")
            return loaded_model
        except Exception as e:
            print(f"Error loading the ARIMA model: {e}")
            return None

    def predict(self, steps, historical_data=None):
        """
        Makes future predictions based on the loaded ARIMA model.
        :param steps: Number of future steps to predict.
        :param historical_data: Optional historical data to update the model before predicting.
        :return: Forecasted values as a numpy array.
        """
        if historical_data is not None:
            self.model = self.model.append(historical_data)

        forecast = self.model.get_forecast(steps=steps)
        forecast_values = forecast.predicted_mean
        return forecast_values

    def evaluate_model(self, actual_values, predicted_values):
        """
        Evaluates the model's performance using Root Mean Squared Error (RMSE).
        :param actual_values: Actual observed values.
        :param predicted_values: Model's predicted values.
        :return: Calculated RMSE.
        """
        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
        print(f"ARIMA Model RMSE: {rmse}")
        return rmse

# Example usage
if __name__ == "__main__":
    arima_service = ARIMAService()
    # Example: predict the next 5 steps
    predictions = arima_service.predict(steps=5)
    print(predictions)
