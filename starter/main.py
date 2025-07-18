# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import pickle
from starter2.ml.data import process_data
from starter2.ml.model import inference
import os

app = FastAPI()

# Load model and encoders

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # directory of the current script (main.py)
model_path = os.path.join(BASE_DIR, "model", "model.pkl")
encoder_path = os.path.join(BASE_DIR, "model", "encoder.pkl")
lb_path = os.path.join(BASE_DIR, "model", "lb.pkl")


with open(model_path, "rb") as f:
    model = pickle.load(f)
with open(encoder_path, "rb") as f:
    encoder = pickle.load(f)
with open(lb_path, "rb") as f:
    lb = pickle.load(f)

# Categorical features
cat_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

# Pydantic model with field aliases for hyphenated names
class CensusInput(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "age": 37,
                "workclass": "Private",
                "fnlgt": 284582,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 45,
                "native-country": "United-States"
            }
        }

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Census Income Prediction API"}

@app.post("/predict")
async def predict(input_data: CensusInput):
    data = input_data.dict(by_alias=True)
    df = pd.DataFrame([data])

    X, _, _, _ = process_data(df, categorical_features=cat_features, training=False, encoder=encoder, lb=lb)
    pred = inference(model, X)
    pred_label = lb.inverse_transform(pred)[0]

    return {"prediction": pred_label}
