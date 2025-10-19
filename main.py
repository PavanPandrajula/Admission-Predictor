import os
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Load models from project root
BASE_DIR = os.path.dirname(__file__)
models = joblib.load(os.path.join(BASE_DIR, 'ensemble_models.pkl'))

class InputData(BaseModel):
    GRE: float
    SOP: float
    PS: float
    LOR: float
    Research: int
    IELTS: int
    Country: str
    Experience: int

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
def predict(data: InputData):
    country_USA = 1 if data.Country.upper() == "USA" else 0
    country_INT = 1 if data.Country.upper() == "INT" else 0
    
    GRE = (data.GRE - 260) / (340-260)
    SOP = (data.SOP - 1) / (5-1)
    PS = (data.PS - 1) / (5-1)
    LOR = (data.LOR - 1) / (5-1)
    
    features = [GRE, SOP, PS, LOR, data.Research, data.IELTS, country_USA, country_INT]
    
    probs = [model.predict_proba([features])[0][1] for model in models.values()]
    avg_prob = np.mean(probs)
    prediction = int(avg_prob >= 0.5)
    
    return {"prediction": prediction, "probability": float(avg_prob)}
