from pydantic import BaseModel, Field
from typing import Literal


class ClientInput(BaseModel):
    util_rate         : float = Field(..., ge=0, le=1,    description="Credit utilization rate (0-1)")
    age               : int   = Field(..., ge=18, le=100, description="Client age")
    late_30_59        : int   = Field(..., ge=0,          description="Times 30-59 days late")
    debt_ratio        : float = Field(..., ge=0,          description="Debt ratio")
    monthly_income    : float = Field(..., ge=0,          description="Monthly income in USD")
    open_credits      : int   = Field(..., ge=0,          description="Number of open credit lines")
    late_90           : int   = Field(..., ge=0,          description="Times 90+ days late")
    real_estate_loans : int   = Field(..., ge=0,          description="Number of real estate loans")
    late_60_89        : int   = Field(..., ge=0,          description="Times 60-89 days late")
    dependents        : int   = Field(..., ge=0,          description="Number of dependents")
    language          : Literal["en", "ar"] = Field("en", description="Report language")

    class Config:
        json_schema_extra = {
            "example": {
                "util_rate"         : 0.75,
                "age"               : 35,
                "late_30_59"        : 1,
                "debt_ratio"        : 0.45,
                "monthly_income"    : 5000,
                "open_credits"      : 5,
                "late_90"           : 0,
                "real_estate_loans" : 1,
                "late_60_89"        : 0,
                "dependents"        : 2,
                "language"          : "en"
            }
        }


class FactorItem(BaseModel):
    feature   : str
    ar_label  : str
    en_label  : str
    direction : Literal["increases", "decreases"]
    impact    : float
    shap_val  : float


class PredictResponse(BaseModel):
    risk_score  : float
    risk_label  : Literal["Low Risk", "Medium Risk", "High Risk"]
    probability : str


class ExplainResponse(BaseModel):
    risk_score  : float
    risk_label  : Literal["Low Risk", "Medium Risk", "High Risk"]
    probability : str
    factors     : list[FactorItem]
    llm_report  : str
    language    : Literal["en", "ar"]