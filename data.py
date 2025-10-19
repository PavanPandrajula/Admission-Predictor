# main.py
import os
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# --------------------
# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ensemble_models.pkl")

# --------------------
# FastAPI app
app = FastAPI(title="Admission Predictor API")

# --------------------
# Generate synthetic data and train models if model doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Training models and creating ensemble_models.pkl...")

    n_rows = 10000
    Country = ["USA", "INT"]

    # Generate features
    Data = pd.DataFrame({
        "Application_number": np.random.randint(100000, 200000, size=n_rows),
        "GRE": np.random.randint(260, 340, size=n_rows),
        "SOP": np.round(np.random.uniform(1, 5, size=n_rows), 1),
        "PS": np.round(np.random.uniform(1, 5, size=n_rows), 1),
        "LOR": np.round(np.random.uniform(1, 5, size=n_rows), 1),
        "Research": np.random.choice([0, 1], size=n_rows),
        "IELTS": np.random.choice([0, 1], size=n_rows),
        "Country": np.random.choice(Country, size=n_rows),
        "Experience": np.random.randint(1, 3, size=n_rows)
    })

    # One-hot encode Country
    encoder = OneHotEncoder(sparse_output=False)
    country_encoded = encoder.fit_transform(Data[['Country']])
    country_columns = encoder.get_feature_names_out(['Country'])
    Data[country_columns] = country_encoded
    Data = Data.drop(columns=['Country'])

    # Scale numeric features
    scaler = MinMaxScaler()
    Data[['GRE', 'PS', 'SOP', 'LOR']] = scaler.fit_transform(Data[['GRE', 'PS', 'SOP', 'LOR']])

    # Target variable
    Data["Admit"] = ((Data["GRE"] + Data["SOP"] + Data["PS"] +
                      Data["Research"] + Data["LOR"] + Data["IELTS"] + Data["Experience"]) > 4.5).astype(int)

    # Features and target
    X = Data[['GRE', 'SOP', 'PS', 'LOR', 'Research', 'IELTS', 'Country_USA', 'Country_INT']]
    y = Data['Admit']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

    # Define models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier()
    }

    # Train models
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name} trained successfully.")

    # Save models
    joblib.dump(models, MODEL_PATH)
    print("Models saved to ensemble_models.pkl")

# --------------------
# Load models
models = joblib.load(MODEL_PATH)

# --------------------
# Input schema for API
class AdmitInput(BaseModel):
    GRE: float
    SOP: float
    PS: float
    LOR: float
    Research: int
    IELTS: int
    Country_USA: int
    Country_INT: int
    Experience: int

# --------------------
# Prediction endpoint
@app.post("/predict")
def predict_admit(input: AdmitInput):
    features = np.array([[
        input.GRE, input.SOP, input.PS, input.LOR,
        input.Research, input.IELTS,
        input.Country_USA, input.Country_INT,
        input.Experience
    ]])

    # Soft voting ensemble
    probabilities = []
    for model in models.values():
        probabilities.append(model.predict_proba(features)[:, 1])

    avg_prob = np.mean(probabilities)
    prediction = int(avg_prob >= 0.5)

    return {
        "admit_probability": float(avg_prob),
        "admit_prediction": prediction
    }
