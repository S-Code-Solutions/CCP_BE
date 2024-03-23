import os

class Config:
    # Secret key for signing cookies, session data, etc.
    # Uncomment and use a real secret key for production
    # SECRET_KEY = os.getenv('SECRET_KEY', 'your_secret_key_here')

    # Paths to machine learning models, with default paths set to the 'model' directory
    ARIMA_MODEL_PATH = os.getenv("ARIMA_MODEL_PATH", "model/arima_model.pkl")
    LSTM_MODEL_PATH = os.getenv("LSTM_MODEL_PATH", "model/lstm_model.pkl")
    PERSONALIZED_PRODUCT_SUGGESTIONS_PATH = os.getenv(
        "PERSONALIZED_PRODUCT_SUGGESTIONS_PATH", "model/Personalized_Product_Suggestions.pkl"
    )
    COLLAB_FILTERING_MODEL_PATH = os.getenv(
        "COLLAB_FILTERING_MODEL_PATH", "model/collab_filtering_model.pkl"
    )
    ASSOCIATION_RULES_PATH = os.getenv(
        "ASSOCIATION_RULES_PATH", "model/association_rules.pkl"
    )

    # Placeholder for future global configuration variables
    # For example, database configurations
    # SQLALCHEMY_DATABASE_URI = 'sqlite:///yourdatabase.db'
    # SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Additional configurations can be added here
