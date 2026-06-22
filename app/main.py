from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pickle
import json
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

from app.schemas import ClientInput, PredictResponse, ExplainResponse

load_dotenv()


# Load model artifacts once at startup

MODEL      = None
EXPLAINER  = None
FEAT_COLS  = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, EXPLAINER, FEAT_COLS

    with open("model/final_model.pkl", "rb") as f:
        MODEL = pickle.load(f)
    with open("model/shap_explainer.pkl", "rb") as f:
        EXPLAINER = pickle.load(f)
    with open("model/feature_cols.json", "r") as f:
        FEAT_COLS = json.load(f)

    print("Model loaded")
    print(" Explainer loaded")
    print(f" Features: {FEAT_COLS}")
    yield
    print("Shutting down...")


app = FastAPI(
    title       = "Smart Loan Risk Analyzer API",
    description = "XGBoost + SHAP + Gemini — Explainable credit risk assessment",
    version     = "1.0.0",
    lifespan    = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


def build_features(data: ClientInput) -> dict:
    d = data.model_dump()

    d["debt_to_income"]       = (d["debt_ratio"] / (d["monthly_income"] + 1)
                                  if d["monthly_income"] > 0 else 0)
    d["total_late"]           = d["late_30_59"] + d["late_60_89"] + d["late_90"]
    d["has_late_history"]     = int(d["total_late"] > 0)
    d["income_per_dependent"] = (d["monthly_income"] / d["dependents"]
                                  if d["dependents"] > 0 else d["monthly_income"])
    d["high_util"]            = int(d["util_rate"] > 0.7)

    return d



@app.get("/", tags=["Health"])
def root():
    return {
        "status"  : "online",
        "model"   : "XGBoost + SHAP + Gemini",
        "version" : "1.0.0",
        "docs"    : "/docs"
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status"      : "healthy",
        "model_loaded": MODEL is not None
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(data: ClientInput):
  
    try:
        features   = build_features(data)
        client_df  = pd.DataFrame([features])[FEAT_COLS]
        risk_score = float(MODEL.predict_proba(client_df)[:, 1][0])

        risk_label = (
            "High Risk"   if risk_score > 0.6 else
            "Medium Risk" if risk_score > 0.3 else
            "Low Risk"
        )

        return PredictResponse(
            risk_score  = round(risk_score, 4),
            risk_label  = risk_label,
            probability = f"{risk_score:.1%}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplainResponse, tags=["Explanation"])
def explain(data: ClientInput):

    try:
        import shap
        from app.llm_explainer import shap_to_factors, explain_client

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="GEMINI_API_KEY not found in environment"
            )

        features = build_features(data)
        result   = explain_client(
            client_dict = features,
            language    = data.language,
            api_key     = api_key
        )

        return ExplainResponse(
            risk_score  = result["risk_score"],
            risk_label  = result["risk_label"],
            probability = f"{result['risk_score']:.1%}",
            factors     = result["factors"],
            llm_report  = result["llm_report"],
            language    = result["language"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))