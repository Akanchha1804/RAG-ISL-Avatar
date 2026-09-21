
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="ISL MediaPipe Landmark Server")


# -----------------------------
# Data Models
# -----------------------------

class Landmark(BaseModel):
    x: float
    y: float
    z: float


class LandmarkData(BaseModel):
    landmarks: List[Landmark]


# -----------------------------
# Temporary Storage
# -----------------------------

latest_landmarks = []


# -----------------------------
# Test Endpoint
# -----------------------------

@app.get("/test")
def test_connection():
    return {
        "pc1_ip": "192.168.1.4",
        "status": "connected"
    }


# -----------------------------
# Receive Landmarks
# -----------------------------

@app.post("/landmarks")
def receive_landmarks(data: LandmarkData):
    global latest_landmarks

    latest_landmarks = [
        landmark.model_dump()
        for landmark in data.landmarks
    ]

    return {
        "status": "received",
        "landmark_count": len(latest_landmarks)
    }


# -----------------------------
# Get Latest Landmarks
# -----------------------------

@app.get("/landmarks")
def get_landmarks():
    return {
        "landmarks": latest_landmarks
    }
