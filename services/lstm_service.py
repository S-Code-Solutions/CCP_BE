import numpy as np
from keras.models import load_model

class LSTMService:
    def __init__(self, model_path='model/lstm_model.h5'):  # Changed extension to .h5
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        """
        Loads the LSTM model from the specified file.
        """
        try:
            loaded_model = load_model(self.model_path)
            print("LSTM model loaded successfully.")
            return loaded_model
        except Exception as e:
            print(f"Error loading the LSTM model: {e}")
            return None

    def predict(self, input_data):
        """
        Makes predictions based on the loaded LSTM model.
        Adjust the reshaping of input_data according to your model's expected input shape.
        """
        # Example reshaping, adjust based on your model's needs
        if len(input_data.shape) == 1:
            input_data = np.expand_dims(input_data, axis=0)  # Handling single feature vector
        predictions = self.model.predict(input_data)
        return predictions.squeeze()

    def preprocess_input(self, data):
        """
        Optional: Preprocess input data before making predictions.
        This might include scaling, reshaping, etc., depending on how the model was trained.
        :param data: The raw input data.
        :return: Preprocessed data ready for prediction.
        """
        # Implement preprocessing steps here
        preprocessed_data = data  # Placeholder, replace with actual preprocessing
        return preprocessed_data

# Example usage
if __name__ == "__main__":
    lstm_service = LSTMService()
    # Dummy input for demonstration, replace with actual data
    dummy_input = np.random.rand(10)  # Example input; ensure this matches the shape your model expects
    preprocessed_input = lstm_service.preprocess_input(dummy_input)
    prediction = lstm_service.predict(preprocessed_input)
    print(prediction)
