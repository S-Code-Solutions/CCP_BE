from fastapi import FastAPI, HTTPException
from typing import List
from .models import UserRequest, RecommendationResponse
from .recommendation import generate_recommendations

app = FastAPI(title="Product Recommendation API", version="1.0")

@app.post("/recommendations/", response_model=List[RecommendationResponse])
async def get_recommendations(request: UserRequest):
    """
    Endpoint to generate product recommendations for a given user.
    Expects a UserRequest model and returns a list of RecommendationResponse models.
    """
    user_id = request.user_id
    # Here we call our recommendation logic.
    recommendations = generate_recommendations(user_id, request.num_recommendations)
    
    if not recommendations:
        raise HTTPException(status_code=404, detail="No recommendations found.")
    return recommendations
