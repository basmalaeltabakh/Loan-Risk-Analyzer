# 🏦 Smart Loan Risk Analyzer

**Explainable AI-powered credit risk assessment system** that predicts the probability of loan default and provides transparent, actionable explanations for every prediction.

Unlike traditional credit scoring models, this solution combines machine learning, explainable AI, and large language models to help financial institutions understand not only the risk level but also the key factors driving the decision.

---

##  Key Features

* **Loan Default Prediction** using XGBoost
* **SHAP Explainability** for feature-level insights
* **AI-Generated Reports** in English and Arabic using Gemini AI
* **Personalized Improvement Roadmap** with actionable recommendations
* **REST API** built with FastAPI (`/predict`, `/explain`)
* **Interactive Dashboard** built with Streamlit and Plotly
* **Risk Visualization** through gauges, feature importance charts, and SHAP plots

---

## 📊 Performance

| Metric              | Score            |
| ------------------- | ---------------- |
| ROC-AUC (Test Set)  | **0.87**         |
| ROC-AUC (5-Fold CV) | **0.84 ± 0.016** |
| Training Samples    | **150,000**      |
| Features            | **15**           |

---

## 🛠️ Tech Stack

* **Machine Learning:** XGBoost, Optuna, SMOTE
* **Explainability:** SHAP (TreeExplainer)
* **LLM Integration:** gemini-3-flash-preview
* **Backend:** FastAPI, Uvicorn
* **Frontend:** Streamlit, Plotly
* **Data Processing:** Pandas, NumPy, Scikit-Learn

---

## 📡 API Endpoints

### POST `/predict`

Returns:

* Risk score
* Default probability
* Risk category (Low / Medium / High)

### POST `/explain`

Returns:

* Prediction result
* SHAP feature contributions
* AI-generated risk report
* Personalized recommendations

---

## 📈 Business Value

* Improves transparency in credit decisions
* Supports explainable and responsible AI practices
* Helps loan officers understand risk drivers
* Provides customers with clear improvement recommendations

---

## 📂 Dataset

**Give Me Some Credit (Kaggle Competition)**

👉 [Dataset Link](https://www.kaggle.com/competitions/GiveMeSomeCredit/data)

- 150,000 loan applicants
- Target: Serious delinquency within 2 years
- Class imbalance handled using SMOTE

---
