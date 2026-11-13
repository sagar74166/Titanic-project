
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import pandas as pd
import joblib

# Create FastAPI application
app = FastAPI()

# Load trained ML model
model = joblib.load("titanic_model.pkl")


# Passenger input validation
class Passenger(BaseModel):

    age: float
    fare: float

    pclass: Literal[1, 2, 3]

    sex: Literal["female", "male"]

    sibsp: Literal[0, 1, 2, 3, 4, 5, 8]

    parch: Literal[0, 1, 2, 3, 4, 5, 6]

    embarked: Literal["C", "Q", "S"]


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Titanic Prediction API is running!"
    }


# Prediction endpoint
@app.post("/predict")
def predict_survival(data: Passenger):

    row = {
        "Age": data.age,
        "Fare": data.fare,

        "Pclass_1": int(data.pclass == 1),
        "Pclass_2": int(data.pclass == 2),
        "Pclass_3": int(data.pclass == 3),

        "Sex_female": int(data.sex == "female"),
        "Sex_male": int(data.sex == "male"),

        **{
            f"SibSp_{i}": int(data.sibsp == i)
            for i in [0, 1, 2, 3, 4, 5, 8]
        },

        **{
            f"Parch_{i}": int(data.parch == i)
            for i in [0, 1, 2, 3, 4, 5, 6]
        },

        "Embarked_C": int(data.embarked == "C"),
        "Embarked_Q": int(data.embarked == "Q"),
        "Embarked_S": int(data.embarked == "S")
    }

    input_df = pd.DataFrame([row])

    prediction = int(model.predict(input_df)[0])

    return {
        "prediction": prediction,
        "result": (
            "Survived" if prediction == 1
            else "Not Survived"
        )
    }
