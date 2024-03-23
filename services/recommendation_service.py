# services/recommendation_service.py

import pandas as pd
import pickle
from typing import List, Union

# Assuming the models are placed directly within a 'models' directory
COLLAB_FILTERING_MODEL_PATH = 'model/collab_filtering_model.pkl'
ASSOCIATION_RULES_MODEL_PATH = 'model/association_rules.pkl'

def load_model(model_path: str):
    """Utility function to load a pickle model."""
    with open(model_path, 'rb') as file:
        return pickle.load(file)

class RecommendationService:
    def __init__(self):
        self.collab_filtering_model = load_model(COLLAB_FILTERING_MODEL_PATH)
        self.association_rules_model = load_model(ASSOCIATION_RULES_MODEL_PATH)

    def get_collab_filtering_recommendations(self, user_id: Union[int, str]) -> List[str]:
        """
        Generates product recommendations for a user based on collaborative filtering model.
        
        Parameters:
            user_id (int or str): The ID of the user for whom to generate recommendations.
        
        Returns:
            List[str]: A list of recommended product titles.
        """
        # Implement the logic to use self.collab_filtering_model to generate recommendations.
        # This is a placeholder implementation.
        recommended_products = ["Product A", "Product B", "Product C"]
        return recommended_products

    def get_association_rule_recommendations(self, basket: List[str]) -> List[str]:
        """
        Generates product recommendations based on the association rules model and a given basket of products.
        
        Parameters:
            basket (List[str]): A list of product titles that are currently in the user's basket.
        
        Returns:
            List[str]: A list of recommended product titles based on association rules.
        """
        # Implement the logic to use self.association_rules_model to generate recommendations.
        # This is a placeholder implementation.
        recommended_products = ["Product D", "Product E"]
        return recommended_products

    def get_recommendations(self, user_id: int, k: int = 10) -> List[Product]:
        """
        Get personalized product recommendations for a given user.
        
        :param user_id: ID of the user for whom to generate recommendations
        :param k: Number of recommendations to generate
        :return: A list of Product instances recommended for the user
        """
        # This is a placeholder implementation
        # Your actual logic will depend on how your model generates recommendations
        # For example, if your model predicts product IDs:
        recommended_product_ids = self.model.predict(user_id, k)

        # Fetch product details from your database or API based on these IDs
        # This is a simplified example; adapt it to your actual data fetching logic
        recommended_products = self.fetch_products(recommended_product_ids)
        return recommended_products

    def fetch_products(self, product_ids: List[int]) -> List[Product]:
        """
        Fetch product details for the given list of product IDs.
        
        :param product_ids: A list of product IDs
        :return: A list of Product instances
        """
        # Placeholder for fetching product data
        # Adapt this method to match how your application's data layer or external product API works
        # Example:
        # products = Product.query.filter(Product.id.in_(product_ids)).all()
        # return products
        return []  # Return an empty list for this placeholder implementation

# The RecommendationService can be instantiated and used across your FastAPI app
