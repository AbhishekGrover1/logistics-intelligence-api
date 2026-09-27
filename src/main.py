from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

# Initialize the FastAPI application
app = FastAPI(
    title="Logistics Intelligence API",
    description="Real-time freight cost prediction engine based on Brazilian e-commerce data.",
    version="1.0.0"
)

# Load the trained machine learning model into memory when the server starts
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "freight_model.joblib")
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Did you run train.py first?")

# Define the strict JSON structure we expect from users
class FreightRequest(BaseModel):
    product_weight_g: float = Field(..., gt=0, description="Weight of the package in grams")
    product_length_cm: float = Field(..., gt=0, description="Length in centimeters")
    product_height_cm: float = Field(..., gt=0, description="Height in centimeters")
    product_width_cm: float = Field(..., gt=0, description="Width in centimeters")
    seller_state: str = Field(..., min_length=2, max_length=2, description="2-letter origin state code (e.g., SP)")
    customer_state: str = Field(..., min_length=2, max_length=2, description="2-letter destination state code (e.g., RJ)")

@app.post("/predict-freight")
def predict_freight(request: FreightRequest):
    try:
        # Convert the incoming JSON payload directly into a Pandas DataFrame
        input_data = pd.DataFrame([request.model_dump()])
        
        # Pass the data through the Scikit-Learn pipeline (scales, encodes, and predicts)
        prediction = model.predict(input_data)[0]
        
        # Return a structured JSON response
        return {
            "status": "success",
            "predicted_freight_value_brl": round(float(prediction), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "online", "model": "Random Forest Regressor"}