
import pickle
import json
import numpy as np
import pandas as pd
import google.generativeai as genai


with open("model/final_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/shap_explainer.pkl", "rb") as f:
    explainer = pickle.load(f)
with open("model/feature_cols.json", "r") as f:
    feature_cols = json.load(f)


def shap_to_factors(shap_vals, feature_names, top_n=5):
    labels = {
        "util_rate"            : ("معدل استخدام الائتمان", "credit utilization rate"),
        "total_late"           : ("عدد مرات التأخر",        "late payment count"),
        "has_late_history"     : ("وجود تاريخ تأخر",        "late payment history"),
        "age"                  : ("العمر",                  "age"),
        "debt_ratio"           : ("نسبة الديون",            "debt ratio"),
        "monthly_income"       : ("الدخل الشهري",           "monthly income"),
        "debt_to_income"       : ("نسبة الدين للدخل",       "debt-to-income ratio"),
        "open_credits"         : ("عدد خطوط الائتمان",      "open credit lines"),
        "late_90"              : ("تأخر أكثر من 90 يوم",    "90+ days late"),
        "late_30_59"           : ("تأخر 30-59 يوم",         "30-59 days late"),
        "late_60_89"           : ("تأخر 60-89 يوم",         "60-89 days late"),
        "income_per_dependent" : ("الدخل لكل معال",         "income per dependent"),
        "high_util"            : ("استخدام ائتمان مرتفع",   "high credit usage"),
        "real_estate_loans"    : ("قروض عقارية",            "real estate loans"),
        "dependents"           : ("عدد المعالين",           "number of dependents"),
    }
    pairs = sorted(
        zip(shap_vals, feature_names),
        key=lambda x: abs(x[0]),
        reverse=True
    )[:top_n]

    factors = []
    for sv, fn in pairs:
        ar, en = labels.get(fn, (fn, fn))
        factors.append({
            "feature"   : fn,
            "ar_label"  : ar,
            "en_label"  : en,
            "direction" : "increases" if sv > 0 else "decreases",
            "impact"    : round(abs(sv), 4),
            "shap_val"  : round(sv, 4)
        })
    return factors


def explain_client(client_dict: dict, language: str = "en",
                   api_key: str = "") -> dict:

    client_df  = pd.DataFrame([client_dict])[feature_cols]
    risk_score = float(model.predict_proba(client_df)[:, 1][0])
    shap_exp   = explainer(client_df)
    factors    = shap_to_factors(shap_exp.values[0], feature_cols)

    risk_label = (
        "High Risk"   if risk_score > 0.6 else
        "Medium Risk" if risk_score > 0.3 else
        "Low Risk"
    )
    risk_label_ar = (
        "مخاطرة عالية"   if risk_score > 0.6 else
        "مخاطرة متوسطة"  if risk_score > 0.3 else
        "مخاطرة منخفضة"
    )

    factors_text_en = "\n".join([
        f"- {f['en_label']}: {f['direction']} default risk (impact: {f['impact']})"
        for f in factors
    ])
    factors_text_ar = "\n".join([
        f"- {f['ar_label']}: {'يزيد' if f['direction'] == 'increases' else 'يقلل'} من مخاطر التعثر (تأثير: {f['impact']})"
        for f in factors
    ])

    if language == "ar":
        prompt = f"""أنت محلل مخاطر ائتمانية محترف في بنك. مهمتك كتابة تقرير واضح وموجز.

درجة المخاطرة: {risk_score:.1%} — {risk_label_ar}

أهم العوامل المؤثرة:
{factors_text_ar}

اكتب تقريراً من 3 فقرات:
1. ملخص وضع العميل والتصنيف النهائي
2. أهم العوامل التي أثّرت في القرار وسببها
3. توصية واضحة للبنك (موافقة / رفض / موافقة بشروط)

اكتب بالعربية الفصحى بأسلوب مهني."""

    else:
        prompt = f"""You are a professional credit risk analyst at a bank.

Default Probability: {risk_score:.1%} — {risk_label}

Key Factors (SHAP):
{factors_text_en}

Write a 3-paragraph report:
1. Executive summary and final classification
2. Key drivers and why they matter
3. Clear recommendation (Approve / Reject / Approve with conditions)

Be professional and data-driven."""

    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel("gemini-3-flash-preview")
    response     = gemini_model.generate_content(prompt)

    return {
        "risk_score" : round(risk_score, 4),
        "risk_label" : risk_label,
        "factors"    : factors,
        "llm_report" : response.text,
        "language"   : language
    }
    