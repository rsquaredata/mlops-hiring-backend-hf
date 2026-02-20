from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
from pymongo import MongoClient

app = FastAPI()

# Load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Mongo connection (via HF Secrets)
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["mlops_db"]
collection = db["predictions"]

class Candidate(BaseModel):
    Age: int
    Gender: int
    EducationLevel: int
    ExperienceYears: int
    PreviousCompanies: int
    DistanceFromCompany: float
    InterviewScore: float
    SkillScore: float
    PersonalityScore: float
    RecruitmentStrategy: int

@app.get("/")
def root():
    return {"message": "Hiring Prediction API running"}

@app.post("/predict")
def predict(candidate: Candidate):
    features = np.array([[ 
        candidate.Age,
        candidate.Gender,
        candidate.EducationLevel,
        candidate.ExperienceYears,
        candidate.PreviousCompanies,
        candidate.DistanceFromCompany,
        candidate.InterviewScore,
        candidate.SkillScore,
        candidate.PersonalityScore,
        candidate.RecruitmentStrategy
    ]])

    features_scaled = scaler.transform(features)
    probability = model.predict_proba(features_scaled)[0][1]

    result = {
        "input": candidate.dict(),
        "hiring_probability": round(float(probability), 4)
    }

    collection.insert_one(result)

    return {"hiring_probability": round(float(probability), 4)}
