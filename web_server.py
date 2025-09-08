from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import joblib

app = FastAPI()
model = joblib.load("regression.joblib")

class ml_response:
    def __init__(self, pred):
        self.y_pred = pred

@app.post("/predict/")
async def predict(size: float = 0.0, bedrooms: float = 0.0, garden: float = 0.0):
    prediction = model.predict([[size, bedrooms, garden]])[0]
    return JSONResponse(content=jsonable_encoder(ml_response(prediction)))