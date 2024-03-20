from fastapi import Header, HTTPException

async def get_token_header(x_token: str = Header(...)):
    """
    A dependency that extracts and validates a token from the request headers.
    """
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")

# This could be used in your main.py like so:
# app.get("/items/", dependencies=[Depends(get_token_header)])

from fastapi import Depends
import pickle

model = None

def load_model():
    """
    Load the model from a .pkl file and store it in a global variable.
    This function could be improved with caching mechanisms or other optimizations.
    """
    global model
    if model is None:
        with open("model/model.pkl", "rb") as file:
            model = pickle.load(file)
    return model

# Usage in an endpoint
# def get_recommendations(user_id: int, model: Any = Depends(load_model)):
#     ...
