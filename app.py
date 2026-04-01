import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize FastAPI app
app = FastAPI(
    title="Smart Health Monitoring API",
    description="API for predicting health status based on vitals using a Random Forest model.",
    version="1.0.0",
)

# Define health status mapping
EXPLAIN_MAP = {
    0: {
        "status": "NORMAL",
        "risk": "Low",
        "analysis": "All vital signs are within clinically acceptable ranges. Signals indicate stable physiology.",
        "recommendation": "Maintain healthy lifestyle and routine monitoring."
    },
    1: {
        "status": "NEAR NORMAL",
        "risk": "Low",
        "analysis": "One parameter slightly deviates but overall condition is stable.",
        "recommendation": "Recheck readings and continue observation."
    },
    2: {
        "status": "MINOR VARIATION",
        "risk": "Low",
        "analysis": "Single mild abnormality detected.",
        "recommendation": "Rest and re-measure."
    },
    3: {
        "status": "MILD IMBALANCE",
        "risk": "Low–Moderate",
        "analysis": "Two mild deviations observed.",
        "recommendation": "Monitor for trend."
    },
    4: {
        "status": "MILD RISK",
        "risk": "Moderate",
        "analysis": "Combination of mild and moderate abnormal signals.",
        "recommendation": "Lifestyle correction advised."
    },
    5: {
        "status": "CARDIAC LOAD SUSPECT",
        "risk": "Moderate",
        "analysis": "Heart rate or ECG strain pattern.",
        "recommendation": "Avoid exertion and monitor ECG."
    },
    6: {
        "status": "RESPIRATORY WATCH",
        "risk": "Moderate",
        "analysis": "SpO2 slightly reduced.",
        "recommendation": "Check breathing and oxygen."
    },
    7: {
        "status": "MULTI-SIGNAL STRESS",
        "risk": "Moderate",
        "analysis": "Multiple signals mildly abnormal.",
        "recommendation": "Continuous monitoring needed."
    },
    8: {
        "status": "HIGH RISK SIGNAL",
        "risk": "High",
        "analysis": "One critical parameter detected.",
        "recommendation": "Immediate retest required."
    },
    9: {
        "status": "CARDIAC WARNING",
        "risk": "High",
        "analysis": "Critical ECG/heart rhythm abnormality.",
        "recommendation": "Cardiac evaluation recommended."
    },
    10: {
        "status": "OXYGEN/FEVER ALERT",
        "risk": "High",
        "analysis": "Critical SpO2 or temperature.",
        "recommendation": "Check oxygen and infection signs."
    },
    11: {
        "status": "ESCALATING INSTABILITY",
        "risk": "High",
        "analysis": "Critical + mild abnormal combination.",
        "recommendation": "Consult doctor."
    },
    12: {
        "status": "SERIOUS CONDITION",
        "risk": "Very High",
        "analysis": "Two critical signals present.",
        "recommendation": "Urgent clinical review."
    },
    13: {
        "status": "CARDIO-RESPIRATORY DANGER",
        "risk": "Very High",
        "analysis": "Heart and oxygen systems abnormal.",
        "recommendation": "Emergency assessment."
    },
    14: {
        "status": "MULTI-SYSTEM RISK",
        "risk": "Critical",
        "analysis": "Three signals critical.",
        "recommendation": "Hospital care advised."
    },
    15: {
        "status": "MEDICAL EMERGENCY",
        "risk": "Critical",
        "analysis": "Most vitals critical.",
        "recommendation": "Immediate emergency intervention."
    }
}

# Input data model
class HealthMetrics(BaseModel):
    age: int = Field(..., example=45, description="Age of the patient")
    heart_rate: int = Field(..., example=80, description="Heart rate in bpm")
    spo2: int = Field(..., example=98, description="Oxygen saturation in %")
    temperature: float = Field(..., example=98.6, description="Body temperature in Fahrenheit")
    ecg: float = Field(..., example=0.5, description="ECG measurement")

class SensorPayload(BaseModel):
    ecg: float
    ir: int
    red: int
    bpm: float
    ambientTemp: float
    objectTemp: float

# Global variables for model and scaler
model = None
scaler = None

@app.on_event("startup")
async def load_model():
    global model, scaler
    try:
        # Load from root or fallback to model/ directory
        model_path = "model.pkl"
        scaler_path = "scaler.pkl"
        
        if not os.path.exists(model_path):
            model_path = os.path.join("model", "health_model.pkl")
            scaler_path = os.path.join("model", "scaler.pkl")
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError("Model loading failed.")

@app.get("/")
def read_root():
    return {"message": "Smart Health Monitoring API is running. Go to /docs for Swagger UI."}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(metrics: HealthMetrics):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Prepare input data
    input_df = pd.DataFrame(
        [[metrics.age, metrics.heart_rate, metrics.spo2, metrics.temperature, metrics.ecg]],
        columns=["age", "heart_rate", "spo2", "temperature", "ecg"]
    )

    try:
        # Apply scaling
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = int(model.predict(input_scaled)[0])

        # Get detailed explanation
        explanation = EXPLAIN_MAP.get(prediction, {
            "status": "Unknown",
            "risk": "Unknown",
            "analysis": "Model output not mapped.",
            "recommendation": "Manual review needed."
        })

        return {
            "prediction": prediction,
            "health_status": explanation["status"],
            "risk_level": explanation["risk"],
            "analysis": explanation["analysis"],
            "recommendation": explanation["recommendation"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/biometric")
def biometric(payload: SensorPayload):
    # temporary placeholder conversion
    age = 25  # hardcoded for now or pass from app
    heart_rate = int(payload.bpm)
    temperature = payload.objectTemp * 9/5 + 32  # convert C to F
    spo2 = 98  # placeholder until real SpO2 is added
    ecg = payload.ecg

    input_df = pd.DataFrame(
        [[age, heart_rate, spo2, temperature, ecg]],
        columns=["age", "heart_rate", "spo2", "temperature", "ecg"]
    )

    input_scaled = scaler.transform(input_df)
    prediction = int(model.predict(input_scaled)[0])

    explanation = EXPLAIN_MAP.get(prediction, {
        "status": "Unknown",
        "risk": "Unknown",
        "analysis": "Model output not mapped.",
        "recommendation": "Manual review needed."
    })

    return {
        "raw_data": payload.dict(),
        "prediction": prediction,
        "health_status": explanation["status"],
        "risk_level": explanation["risk"],
        "analysis": explanation["analysis"],
        "recommendation": explanation["recommendation"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
