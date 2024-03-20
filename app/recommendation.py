import pickle
from typing import List
from .models import ItemRecommendation

# Path to your trained model .pkl file
MODEL_PATH = "model/Personalized_Product_Suggestions.pkl"


# Load the trained model from a .pkl file
def load_model():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    return model


model = load_model()


# Mock function to simulate model prediction
# Replace this with your actual model prediction logic
def predict(user_id: int, num_recommendations: int) -> List[dict]:
    # Assuming the model returns a list of (item_id, score) tuples
    # This is a placeholder for your model's prediction logic
    predictions = [
        {"item_id": "item123", "item_name": "Product XYZ", "score": 0.95},
        {"item_id": "item456", "item_name": "Product ABC", "score": 0.90},
        # Add more predictions as needed
    ]
    return predictions[:num_recommendations]


# Generate recommendations using the model
def generate_recommendations(
    user_id: int, num_recommendations: int = 5
) -> List[ItemRecommendation]:
    # Call the model prediction function
    predictions = predict(user_id, num_recommendations)

    # Format the predictions as a list of ItemRecommendation objects
    recommendations = [ItemRecommendation(**prediction) for prediction in predictions]
    return recommendations
