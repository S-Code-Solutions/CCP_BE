from pydantic import BaseModel, Field
from typing import List, Optional

class UserRequest(BaseModel):
    user_id: int = Field(..., description="The unique identifier of the user")
    num_recommendations: Optional[int] = Field(5, description="Number of recommendations to fetch")

class ItemRecommendation(BaseModel):
    item_id: str = Field(..., description="The unique identifier of the recommended item")
    item_name: str = Field(..., description="The name of the recommended item")
    score: Optional[float] = Field(None, description="The recommendation score or relevance")

class RecommendationResponse(BaseModel):
    user_id: int = Field(..., description="The unique identifier of the user")
    recommendations: List[ItemRecommendation] = Field(..., description="List of recommended items")
