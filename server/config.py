import os

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'your_secret_key_here')
    # Assuming the .pkl files are in the 'models' folder in the root directory of your project
    ARIMA_MODEL_PATH = os.getenv('ARIMA_MODEL_PATH', 'models/arima_model.pkl')
    LSTM_MODEL_PATH = os.getenv('LSTM_MODEL_PATH', 'models/lstm_model.pkl')
    # Add other global configuration variables here

    # For example, if you had a database
    # SQLALCHEMY_DATABASE_URI = 'sqlite:///yourdatabase.db'
    # SQLALCHEMY_TRACK_MODIFICATIONS = False
