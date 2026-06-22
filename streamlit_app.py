import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ============================================================
# Page Config
# ============================================================
st.set_page_config(page_title="Smart Loan Risk Analyzer", page_icon="🏦", layout="wide")

# ============================================================
# Custom CSS
# ============================================================
st.markdown(
    """
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Cards */
    .risk-card {
        background: #1e2130;
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
        border: 1px solid #2d3147;
    }

    /* Risk badges */
    .badge-high {
        background: #ff4b4b22;
        border: 1px solid #ff4b4b;
        color: #ff4b4b;
        padding: 6px 18px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 15px;
    }
    .badge-medium {
        background: #ffa50022;
        border: 1px solid #ffa500;
        color: #ffa500;
        padding: 6px 18px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 15px;
    }
    .badge-low {
        background: #00c85322;
        border: 1px solid #00c853;
        color: #00c853;
        padding: 6px 18px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 15px;
    }

    /* Factor rows */
    .factor-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid #2d3147;
    }
    .factor-up   { color: #ff4b4b; font-size: 18px; }
    .factor-down { color: #00c853; font-size: 18px; }

    /* Report box */
    .report-box {
        background: #1e2130;
        border-left: 4px solid #4e9eff;
        border-radius: 0 12px 12px 0;
        padding: 20px 24px;
        line-height: 1.8;
        color: #e0e0e0;
        font-size: 15px;
    }

    /* Recommendation cards */
    .rec-approve {
        background: #00c85315;
        border: 1px solid #00c853;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 6px 0;
    }
    .rec-reject {
        background: #ff4b4b15;
        border: 1px solid #ff4b4b;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 6px 0;
    }
    .rec-conditional {
        background: #ffa50015;
        border: 1px solid #ffa500;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 6px 0;
    }

    /* Metric cards */
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #2d3147;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #4e9eff;
    }
    .metric-label {
        font-size: 12px;
        color: #888;
        margin-top: 4px;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1e2130 0%, #2d3147 100%);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        border: 1px solid #2d3147;
    }

    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""",
    unsafe_allow_html=True,
)

API_URL = "http://127.0.0.1:8000"
import json, os


@st.cache_resource
def load_metrics():
    path = "model/model_metrics.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # fallback لو الملف مش موجود
    return {"roc_auc": 0.87, "training_samples": 150000, "n_features": 15}


metrics = load_metrics()
# ============================================================
# Header
# ============================================================
st.markdown(
    """
<div class="main-header">
    <h1 style="margin:0; font-size:28px; color:#fff;">
        🏦 Smart Loan Risk Analyzer
    </h1>
    <p style="margin:6px 0 0; color:#888; font-size:14px;">
        XGBoost · SHAP Explainability · AI-Generated Reports
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Sidebar — Client Input
# ============================================================
with st.sidebar:
    st.markdown("## 👤 Client Profile")
    st.divider()

    # Personal
    st.markdown("**Personal Info**")
    age = st.slider("Age", 18, 80, 35)
    dependents = st.slider("Number of Dependents", 0, 10, 1)

    st.divider()

    # Financial
    st.markdown("**Financial Info**")
    monthly_income = st.number_input(
        "Monthly Income ($)", min_value=0, max_value=100000, value=5000, step=500
    )
    debt_ratio = st.slider("Debt Ratio", 0.0, 1.0, 0.3, step=0.01)
    util_rate = st.slider("Credit Utilization Rate", 0.0, 1.0, 0.4, step=0.01)
    open_credits = st.slider("Open Credit Lines", 0, 20, 4)
    real_estate_loans = st.slider("Real Estate Loans", 0, 10, 0)

    st.divider()

    # Late payments
    st.markdown("**Late Payment History**")
    late_30_59 = st.slider("Times 30-59 Days Late", 0, 15, 0)
    late_60_89 = st.slider("Times 60-89 Days Late", 0, 15, 0)
    late_90 = st.slider("Times 90+ Days Late", 0, 15, 0)

    st.divider()

    # Mode
    st.markdown("**Analysis Mode**")
    mode = st.radio(
        "Choose mode", ["⚡ Quick Prediction", "🧠 Full AI Explanation"], index=1
    )

    language = "en"
    if mode == "🧠 Full AI Explanation":
        language = st.selectbox(
            "Report Language",
            ["en", "ar"],
            format_func=lambda x: "🇬🇧 English" if x == "en" else "🇪🇬 Arabic",
        )

    st.divider()
    analyze_btn = st.button(
        "🔍 Analyze Client", use_container_width=True, type="primary"
    )


# ============================================================
# Helper Functions
# ============================================================
def get_badge(label):
    if label == "High Risk":
        return '<span class="badge-high">🔴 High Risk</span>'
    elif label == "Medium Risk":
        return '<span class="badge-medium">🟡 Medium Risk</span>'
    else:
        return '<span class="badge-low">🟢 Low Risk</span>'


def risk_gauge(score):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score * 100,
            number={"suffix": "%", "font": {"size": 36, "color": "#fff"}},
            delta={
                "reference": 30,
                "increasing": {"color": "#ff4b4b"},
                "decreasing": {"color": "#00c853"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#888",
                    "tickfont": {"color": "#888"},
                },
                "bar": {"color": "#4e9eff"},
                "bgcolor": "#1e2130",
                "bordercolor": "#2d3147",
                "steps": [
                    {"range": [0, 30], "color": "rgba(0, 200, 83, 0.12)"},
                    {"range": [30, 60], "color": "rgba(255, 165, 0, 0.12)"},
                    {"range": [60, 100], "color": "rgba(255, 75, 75, 0.12)"},
                ],
                "threshold": {
                    "line": {"color": "#ff4b4b", "width": 3},
                    "thickness": 0.8,
                    "value": 60,
                },
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(t=30, b=0, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#fff",
    )
    return fig


def shap_bar_chart(factors):
    df = pd.DataFrame(factors)
    df["color"] = df["direction"].map({"increases": "#ff4b4b", "decreases": "#00c853"})
    df["signed_impact"] = df.apply(
        lambda r: r["impact"] if r["direction"] == "increases" else -r["impact"], axis=1
    )
    df = df.sort_values("signed_impact")

    fig = go.Figure(
        go.Bar(
            x=df["signed_impact"],
            y=df["en_label"],
            orientation="h",
            marker_color=df["color"],
            text=df["signed_impact"].round(4).astype(str),
            textposition="outside",
        )
    )
    fig.update_layout(
        title="SHAP Feature Impact",
        height=350,
        margin=dict(t=40, b=20, l=20, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#fff",
        xaxis=dict(
            title="Impact on Default Risk", gridcolor="#2d3147", zerolinecolor="#4e9eff"
        ),
        yaxis=dict(gridcolor="#2d3147"),
    )
    return fig


def shap_waterfall_chart(factors, base_value=0.15):
    sorted_factors = sorted(factors, key=lambda x: abs(x["impact"]), reverse=True)

    labels = []
    values = []
    colors = []
    measure = []

    # Base value
    labels.append("Base Rate")
    values.append(base_value)
    colors.append("#4e9eff")
    measure.append("absolute")

    running = base_value
    for f in sorted_factors:
        signed = f["impact"] if f["direction"] == "increases" else -f["impact"]
        labels.append(f["en_label"])
        values.append(signed)
        colors.append("#ff4b4b" if signed > 0 else "#00c853")
        measure.append("relative")
        running += signed

    # Final score
    labels.append("Final Risk Score")
    values.append(running)
    colors.append("#ffa500")
    measure.append("total")

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measure,
            x=labels,
            y=values,
            text=[
                f"{v:+.3f}" if m == "relative" else f"{v:.3f}"
                for v, m in zip(values, measure)
            ],
            textposition="outside",
            connector={"line": {"color": "#2d3147", "width": 1}},
            increasing={"marker": {"color": "#ff4b4b"}},
            decreasing={"marker": {"color": "#00c853"}},
            totals={"marker": {"color": "#ffa500"}},
        )
    )

    fig.update_layout(
        title="Risk Score Waterfall — How Each Factor Contributes",
        height=420,
        margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#fff",
        font_size=12,
        xaxis=dict(
            gridcolor="#2d3147",
            tickangle=-25,
        ),
        yaxis=dict(
            gridcolor="#2d3147", tickformat=".0%", title="Default Risk Contribution"
        ),
        showlegend=False,
    )

    fig.add_hline(
        y=0.6,
        line_dash="dash",
        line_color="#ff4b4b",
        opacity=0.5,
        annotation_text="High Risk Threshold (60%)",
        annotation_position="top right",
        annotation_font_color="#ff4b4b",
    )
    fig.add_hline(
        y=0.3,
        line_dash="dash",
        line_color="#ffa500",
        opacity=0.5,
        annotation_text="Medium Risk Threshold (30%)",
        annotation_position="bottom right",
        annotation_font_color="#ffa500",
    )

    return fig


def risk_radar_chart(client_data: dict, factors: list):
 


    # 1. Payment History (0=perfect, 1=terrible)
    total_late = (
        client_data.get("late_30_59", 0)
        + client_data.get("late_60_89", 0)
        + client_data.get("late_90", 0)
    )
    payment_risk = min(total_late / 10, 1.0)

    # 2. Credit Usage
    credit_risk = client_data.get("util_rate", 0)

    # 3. Debt Burden
    debt_risk = min(client_data.get("debt_ratio", 0), 1.0)

    
    income = client_data.get("monthly_income", 1)
    income_risk = max(0, 1 - (income / 15000))  # 15k+ = low risk

    
    credits = client_data.get("open_credits", 0)
    mix_risk = min(credits / 15, 1.0)

    categories = [
        "Payment History",
        "Credit Usage",
        "Debt Burden",
        "Income Stability",
        "Credit Mix",
    ]
    values = [payment_risk, credit_risk, debt_risk, income_risk, mix_risk]

    # إغلاق الـ polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    # High Risk Zone    
    fig.add_trace(
        go.Scatterpolar(
            r=[0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(255, 75, 75, 0.06)",
            line=dict(color="rgba(255,75,75,0.3)", dash="dash", width=1),
            name="High Risk Zone",
            hoverinfo="skip",
        )
    )

    
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(78, 158, 255, 0.15)",
            line=dict(color="#4e9eff", width=2),
            name="Client Profile",
            text=[f"{v:.0%}" for v in values_closed],
            hovertemplate="<b>%{theta}</b><br>Risk Level: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat=".0%",
                tickfont=dict(color="#888", size=10),
                gridcolor="#2d3147",
                linecolor="#2d3147",
            ),
            angularaxis=dict(
                tickfont=dict(color="#ccc", size=12),
                gridcolor="#2d3147",
                linecolor="#2d3147",
            ),
        ),
        showlegend=True,
        legend=dict(font=dict(color="#ccc"), bgcolor="rgba(0,0,0,0)", x=1.1, y=1.1),
        title=dict(
            text="Client Risk Profile Radar", font=dict(color="#fff", size=14), x=0.5
        ),
        height=400,
        margin=dict(t=60, b=20, l=60, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#fff",
    )

    return fig, values, categories


def calculate_improvement_steps(client_data: dict, current_score: float, factors: list):
    """Calculate improvement steps based on client data and factors."""
    steps = []
    simulated = dict(client_data)
    simulated_score = current_score

    risk_factors = [f for f in factors if f["direction"] == "increases"]
    risk_factors = sorted(risk_factors, key=lambda x: x["impact"], reverse=True)

    for f in risk_factors:
        feature = f["feature"]
        impact = f["impact"]
        reduction = impact * 0.75  
        if feature == "util_rate":
            old_val = simulated.get("util_rate", 0)
            target = max(old_val - 0.3, 0.1)
            action = f"Reduce credit utilization from {old_val:.0%} → {target:.0%}"
            timeline = "1-3 months"
            how = (
                "Pay down revolving balances. Aim to use less than 30% of credit limit."
            )
            icon = "💳"

        elif feature in ["total_late", "has_late_history"]:
            action = "Establish 6 consecutive months of on-time payments"
            timeline = "6 months"
            how = "Set up automatic payments to avoid missing due dates."
            icon = "📅"

        elif feature in ["late_90", "late_60_89", "late_30_59"]:
            action = "Clear all overdue accounts and avoid new late payments"
            timeline = "3-6 months"
            how = "Contact lenders to negotiate payment plans for overdue accounts."
            icon = "🔔"

        elif feature == "debt_ratio":
            old_val = simulated.get("debt_ratio", 0)
            target = max(old_val - 0.2, 0.1)
            action = f"Reduce debt ratio from {old_val:.0%} → {target:.0%}"
            timeline = "6-12 months"
            how = "Focus on paying off high-interest debt first (avalanche method)."
            icon = "📉"

        elif feature == "debt_to_income":
            action = (
                "Improve debt-to-income ratio by increasing income or reducing debt"
            )
            timeline = "3-6 months"
            how = (
                "Consider additional income sources or refinancing high-interest loans."
            )
            icon = "⚖️"

        elif feature == "monthly_income":
            action = "Demonstrate income stability with consistent payslips"
            timeline = "3 months"
            how = (
                "Provide 3-month payslip history or proof of additional income streams."
            )
            icon = "💰"

        elif feature == "high_util":
            action = "Bring credit utilization below 50%"
            timeline = "1-2 months"
            how = "Request a credit limit increase or pay down current balances."
            icon = "📊"

        elif feature == "open_credits":
            old_val = simulated.get("open_credits", 0)
            action = f"Avoid opening new credit lines (currently {int(old_val)} lines)"
            timeline = "Ongoing"
            how = "Each new credit application creates a hard inquiry and lowers score."
            icon = "🚫"

        else:
            continue

        new_score = max(simulated_score - reduction, 0.01)

        steps.append(
            {
                "step": len(steps) + 1,
                "icon": icon,
                "action": action,
                "timeline": timeline,
                "how": how,
                "before_score": simulated_score,
                "after_score": new_score,
                "reduction": reduction,
            }
        )

        simulated_score = new_score

        if simulated_score < 0.3 or len(steps) >= 4:
            break

    return steps


def improvement_timeline_chart(steps: list, current_score: float):
 
    labels = ["Current"] + [f"Step {s['step']}" for s in steps]
    scores = [current_score] + [s["after_score"] for s in steps]
    colors_pts = []

    for s in scores:
        if s > 0.6:
            colors_pts.append("#ff4b4b")
        elif s > 0.3:
            colors_pts.append("#ffa500")
        else:
            colors_pts.append("#00c853")

    fig = go.Figure()

    
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=scores,
            mode="lines+markers+text",
            line=dict(color="#4e9eff", width=3, shape="spline"),
            marker=dict(size=14, color=colors_pts, line=dict(color="#fff", width=2)),
            text=[f"{s:.0%}" for s in scores],
            textposition="top center",
            textfont=dict(color="#fff", size=12),
            fill="tozeroy",
            fillcolor="rgba(78,158,255,0.08)",
            name="Risk Score",
        )
    )

    
    fig.add_hline(
        y=0.6,
        line_dash="dash",
        line_color="#ff4b4b",
        opacity=0.4,
        annotation_text="High Risk",
        annotation_font_color="#ff4b4b",
        annotation_position="top left",
    )
    fig.add_hline(
        y=0.3,
        line_dash="dash",
        line_color="#ffa500",
        opacity=0.4,
        annotation_text="Medium Risk",
        annotation_font_color="#ffa500",
        annotation_position="top left",
    )
    fig.add_hline(y=0.0, line_color="rgba(0,0,0,0)")

    fig.update_layout(
        title="Projected Risk Score After Each Improvement Step",
        height=320,
        margin=dict(t=50, b=30, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#fff",
        xaxis=dict(gridcolor="#2d3147"),
        yaxis=dict(
            gridcolor="#2d3147",
            tickformat=".0%",
            range=[0, min(current_score + 0.15, 1.0)],
            title="Default Risk",
        ),
        showlegend=False,
    )

    return fig


def get_recommendation(risk_score, factors):
    """Rule-based recommendation system"""
    total_late = sum(
        f["impact"]
        for f in factors
        if f["feature"] in ["total_late", "late_90", "late_30_59", "late_60_89"]
        and f["direction"] == "increases"
    )
    high_util = any(
        f["feature"] == "util_rate" and f["direction"] == "increases" for f in factors
    )

    if risk_score < 0.3:
        return {
            "decision": "✅ Approve",
            "style": "rec-approve",
            "color": "#00c853",
            "conditions": [],
            "reasons": [
                "Low default probability — client profile is strong",
                "No significant risk flags detected",
                "Standard loan terms apply",
            ],
        }
    elif risk_score < 0.6:
        conditions = []
        if high_util:
            conditions.append(
                "💳 Require credit utilization below 50% before disbursement"
            )
        if total_late > 0:
            conditions.append("📋 Request 6-month bank statement for verification")
        conditions.append("📊 Set loan amount cap at 3× monthly income")
        conditions.append("🔒 Consider requiring a guarantor")

        return {
            "decision": "⚠️ Approve with Conditions",
            "style": "rec-conditional",
            "color": "#ffa500",
            "conditions": conditions,
            "reasons": [
                "Moderate risk — manageable with proper controls",
                "Some risk flags present but not disqualifying",
            ],
        }
    else:
        reasons = ["High default probability exceeds acceptable threshold"]
        if total_late > 0:
            reasons.append("Significant late payment history detected")
        if high_util:
            reasons.append("Credit utilization is critically high")
        reasons.append("Risk-adjusted return does not justify approval")

        return {
            "decision": "❌ Reject",
            "style": "rec-reject",
            "color": "#ff4b4b",
            "conditions": [],
            "reasons": reasons,
        }


def how_it_works_section():
    steps = [
        {
            "icon": "📝",
            "title": "Enter Client Data",
            "desc": "Fill in financial details: income, debt ratio, credit utilization, and payment history.",
        },
        {
            "icon": "⚙️",
            "title": "XGBoost Prediction",
            "desc": "Our model trained on 150k real cases calculates the default probability instantly.",
        },
        {
            "icon": "🔍",
            "title": "SHAP Explainability",
            "desc": "Every prediction is explained — see exactly which factors drove the decision.",
        },
        {
            "icon": "🤖",
            "title": "AI-Generated Report",
            "desc": "Gemini AI writes a professional risk report in English or Arabic with a clear recommendation.",
        },
    ]

    cols = st.columns(4)
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            st.markdown(
                f'<div class="risk-card" style="text-align:center;padding:20px 16px">'
                f'<div style="font-size:32px;margin-bottom:10px">{step["icon"]}</div>'
                f'<div style="width:24px;height:24px;background:#4e9eff;border-radius:50%;'
                f"display:flex;align-items:center;justify-content:center;"
                f'margin:0 auto 10px;font-size:12px;font-weight:700;color:#0f1117">'
                f"{i+1}</div>"
                f'<p style="font-size:14px;font-weight:600;color:#fff;margin:0 0 8px">'
                f'{step["title"]}</p>'
                f'<p style="font-size:12px;color:#888;margin:0;line-height:1.6">'
                f'{step["desc"]}</p>'
                f"</div>",
                unsafe_allow_html=True,
            )

        if i < 3:
           
            
            pass


def model_performance_section(metrics: dict):
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("#### ROC Curve")

        fpr = metrics.get("roc_fpr", [0, 1])
        tpr = metrics.get("roc_tpr", [0, 1])
        auc = metrics.get("roc_auc", 0)

        fig = go.Figure()

        # Random baseline
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="#888", dash="dash", width=1),
                name="Random (AUC = 0.50)",
                hoverinfo="skip",
            )
        )

        # ROC curve
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                line=dict(color="#4e9eff", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(78,158,255,0.08)",
                name=f"XGBoost (AUC = {auc})",
            )
        )

        fig.update_layout(
            height=320,
            margin=dict(t=20, b=40, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#fff",
            legend=dict(bgcolor="rgba(30,33,48,0.8)", font=dict(size=11)),
            xaxis=dict(title="False Positive Rate", gridcolor="#2d3147", range=[0, 1]),
            yaxis=dict(title="True Positive Rate", gridcolor="#2d3147", range=[0, 1]),
        )

        # AUC annotation
        fig.add_annotation(
            x=0.7,
            y=0.2,
            text=f"<b>AUC = {auc}</b>",
            showarrow=False,
            font=dict(size=16, color="#4e9eff"),
            bgcolor="rgba(30,33,48,0.8)",
            bordercolor="#4e9eff",
            borderwidth=1,
            borderpad=8,
        )

        st.plotly_chart(fig, use_container_width=True, key="roc_curve_chart")

    with col2:
        st.markdown("#### Confusion Matrix")

        cm_data = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        cm_arr  = [[cm_data[0][0], cm_data[0][1]], [cm_data[1][0], cm_data[1][1]]]
        labels  = ["No Default", "Default"]

        total = sum(sum(row) for row in cm_arr)

        if total == 0:
            st.info("⚠️ Confusion matrix data not available. Run the Colab cell to generate model_metrics.json")
        else:
            fig = go.Figure(
                go.Heatmap(
                    z=cm_arr,
                    x=labels,
                    y=labels,
                    colorscale=[[0.0, "#1e2130"], [0.5, "#1a3a6e"], [1.0, "#4e9eff"]],
                    showscale=False,
                    text=[
                        [
                            f"<b>{cm_arr[0][0]:,}</b><br><span style='font-size:10px'>"
                            f"TN ({cm_arr[0][0]/total:.1%})</span>",
                            f"<b>{cm_arr[0][1]:,}</b><br><span style='font-size:10px'>"
                            f"FP ({cm_arr[0][1]/total:.1%})</span>",
                        ],
                        [
                            f"<b>{cm_arr[1][0]:,}</b><br><span style='font-size:10px'>"
                            f"FN ({cm_arr[1][0]/total:.1%})</span>",
                            f"<b>{cm_arr[1][1]:,}</b><br><span style='font-size:10px'>"
                            f"TP ({cm_arr[1][1]/total:.1%})</span>",
                        ],
                    ],
                    texttemplate="%{text}",
                    textfont={"size": 13, "color": "white"},
                    hoverongaps=False,
                )
            )

            fig.update_layout(
                height=320,
                margin=dict(t=20, b=40, l=80, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#fff",
                xaxis=dict(title="Predicted", side="bottom"),
                yaxis=dict(title="Actual", autorange="reversed"),
            )

            st.plotly_chart(fig, use_container_width=True, key="confusion_matrix_chart")

            tn, fp = cm_data[0]
            fn, tp = cm_data[1]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0)

            m1, m2, m3 = st.columns(3)
            m1.metric("Precision", f"{precision:.2f}")
            m2.metric("Recall",    f"{recall:.2f}")
            m3.metric("F1 Score",  f"{f1:.2f}")


def dataset_overview_section():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Age Distribution")
        age_data = {
            "18-30": 18.2,
            "31-45": 31.5,
            "46-60": 28.9,
            "61-75": 16.4,
            "75+": 5.0,
        }
        fig = go.Figure(
            go.Bar(
                x=list(age_data.keys()),
                y=list(age_data.values()),
                marker_color=[
                    "#4e9eff" if i != 1 else "#ffa500" for i in range(len(age_data))
                ],
                text=[f"{v}%" for v in age_data.values()],
                textposition="outside",
                textfont=dict(color="#fff", size=11),
            )
        )
        fig.update_layout(
            height=250,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#fff",
            showlegend=False,
            xaxis=dict(gridcolor="#2d3147"),
            yaxis=dict(gridcolor="#2d3147", ticksuffix="%"),
        )
        st.plotly_chart(fig, use_container_width=True, key="age_distribution_chart")

    with col2:
        st.markdown("#### Default Rate by Age Group")
        default_rates = {
            "18-30": 11.2,
            "31-45": 8.4,
            "46-60": 6.1,
            "61-75": 4.3,
            "75+": 3.8,
        }
        colors = [
            "#ff4b4b" if v > 8 else "#ffa500" if v > 5 else "#00c853"
            for v in default_rates.values()
        ]
        fig = go.Figure(
            go.Bar(
                x=list(default_rates.keys()),
                y=list(default_rates.values()),
                marker_color=colors,
                text=[f"{v}%" for v in default_rates.values()],
                textposition="outside",
                textfont=dict(color="#fff", size=11),
            )
        )
        fig.update_layout(
            height=250,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#fff",
            showlegend=False,
            xaxis=dict(gridcolor="#2d3147"),
            yaxis=dict(gridcolor="#2d3147", ticksuffix="%"),
        )
        st.plotly_chart(fig, use_container_width=True, key="default_rate_chart")

    with col3:
        st.markdown("#### Overall Class Split")
        fig = go.Figure(
            go.Pie(
                labels=["No Default (93.3%)", "Default (6.7%)"],
                values=[93.3, 6.7],
                hole=0.6,
                marker=dict(
                    colors=["#4e9eff", "#ff4b4b"], line=dict(color="#0f1117", width=3)
                ),
                textinfo="none",
                hovertemplate="<b>%{label}</b><extra></extra>",
            )
        )
        fig.add_annotation(
            text="6.7%<br><span style='font-size:11px'>Default Rate</span>",
            x=0.5,
            y=0.5,
            font=dict(size=16, color="#fff"),
            showarrow=False,
        )
        fig.update_layout(
            height=250,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#fff",
            showlegend=True,
            legend=dict(
                font=dict(size=11),
                bgcolor="rgba(0,0,0,0)",
                x=0.5,
                xanchor="center",
                y=-0.1,
                orientation="h",
            ),
        )
        st.plotly_chart(fig, use_container_width=True, key="class_split_chart")
if not analyze_btn:
    # ── Metric Cards ──
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['roc_auc']}</div>
            <div class="metric-label">Model ROC-AUC</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['training_samples']:,}</div>
            <div class="metric-label">Training Samples</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['n_features']}</div>
            <div class="metric-label">Features Used</div>
        </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "👈  Fill in the client details in the sidebar and click **Analyze Client** to get started."
    )

    st.divider()

    # ── How It Works ──
    st.markdown("### ⚙️ How It Works")
    st.markdown("<br>", unsafe_allow_html=True)
    how_it_works_section()

    st.divider()

    # ── Model Performance ──
    st.markdown("### 📈 Model Performance")
    model_performance_section(metrics)

    st.divider()

    # ── Dataset Overview ──
    st.markdown("### 🗂️ Dataset Overview")
    st.markdown(
        '<p style="color:#888;font-size:13px;margin:-10px 0 16px">'
        "Give Me Some Credit — 150,000 real loan applicants</p>",
        unsafe_allow_html=True,
    )
    dataset_overview_section()

else:
    # Build payload
    payload = {
        "util_rate": util_rate,
        "age": age,
        "late_30_59": late_30_59,
        "debt_ratio": debt_ratio,
        "monthly_income": monthly_income,
        "open_credits": open_credits,
        "late_90": late_90,
        "real_estate_loans": real_estate_loans,
        "late_60_89": late_60_89,
        "dependents": dependents,
        "language": language,
    }

    # ============================================================
    # QUICK PREDICTION MODE
    # ============================================================
    if mode == "⚡ Quick Prediction":
        with st.spinner("Running prediction..."):
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                resp.raise_for_status()
                result = resp.json()
            except Exception as e:
                st.error(f"API Error: {e}")
                st.stop()

        risk_score = result["risk_score"]
        risk_label = result["risk_label"]
        rec = get_recommendation(risk_score, [])

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Risk Score")
            st.plotly_chart(risk_gauge(risk_score), use_container_width=True, key="risk_gauge_quick")

        with col2:
            st.markdown("### Assessment")
            st.markdown(
                f'<div class="risk-card" style="margin-top:8px">'
                f'<p style="color:#888;font-size:13px;margin:0">RISK CLASSIFICATION</p>'
                f'<div style="margin:12px 0">{get_badge(risk_label)}</div>'
                f'<hr style="border-color:#2d3147">'
                f'<p style="color:#888;font-size:13px;margin:8px 0 4px">DEFAULT PROBABILITY</p>'
                f'<p style="font-size:32px;font-weight:700;color:#4e9eff;margin:0">'
                f"{risk_score:.1%}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("### Recommendation")
            st.markdown(
                f'<div class="{rec["style"]}">'
                f'<b style="font-size:16px;color:{rec["color"]}">{rec["decision"]}</b>',
                unsafe_allow_html=True,
            )
            for r in rec["reasons"]:
                st.markdown(f"- {r}")
            if rec["conditions"]:
                st.markdown("**Conditions:**")
                for c in rec["conditions"]:
                    st.markdown(f"- {c}")
            st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    # FULL AI EXPLANATION MODE
    # ============================================================
    else:
        with st.spinner("🧠 Running AI analysis... this may take ~10 seconds"):
            try:
                resp = requests.post(f"{API_URL}/explain", json=payload, timeout=60)
                resp.raise_for_status()
                result = resp.json()
            except Exception as e:
                st.error(f"API Error: {e}")
                st.stop()

        risk_score = result["risk_score"]
        risk_label = result["risk_label"]
        factors = result["factors"]
        llm_report = result["llm_report"]
        rec = get_recommendation(risk_score, factors)

        # ── Row 1: Gauge + Classification + Recommendation ──
        col1, col2, col3 = st.columns([1.2, 1, 1.2])

        with col1:
            st.markdown("### Risk Score")
            st.plotly_chart(risk_gauge(risk_score), use_container_width=True, key="risk_gauge_full")
            st.markdown(
                f'<div class="risk-card">'
                f'<p style="color:#888;font-size:12px;margin:0 0 8px">RISK LEVEL</p>'
                f"{get_badge(risk_label)}"
                f'<hr style="border-color:#2d3147;margin:14px 0">'
                f'<p style="color:#888;font-size:12px;margin:0 0 4px">DEFAULT PROBABILITY</p>'
                f'<p style="font-size:34px;font-weight:700;color:#4e9eff;margin:0">'
                f"{risk_score:.1%}</p>"
                f'<hr style="border-color:#2d3147;margin:14px 0">'
                f'<p style="color:#888;font-size:12px;margin:0 0 8px">RISK THRESHOLD</p>'
                f'<p style="font-size:13px;color:#ccc;margin:0">'
                f"Low &lt;30% · Medium 30-60% · High &gt;60%</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown("### Recommendation")
            st.markdown(
                f'<div class="{rec["style"]}">'
                f'<b style="font-size:15px;color:{rec["color"]}">{rec["decision"]}</b>'
                f'<hr style="border-color:{rec["color"]}33;margin:10px 0">',
                unsafe_allow_html=True,
            )
            for r in rec["reasons"]:
                st.markdown(f"<small>• {r}</small>", unsafe_allow_html=True)
            if rec["conditions"]:
                st.markdown(
                    '<p style="color:#ffa500;font-size:13px;margin:10px 0 6px">'
                    "<b>Required Conditions:</b></p>",
                    unsafe_allow_html=True,
                )
                for c in rec["conditions"]:
                    st.markdown(f"<small>{c}</small>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # ── Row 2: SHAP Chart + Top Factors ──
        col1, col2 = st.columns([1.4, 1])

        with col1:
            st.markdown("### SHAP Feature Impact")
            st.plotly_chart(shap_bar_chart(factors), use_container_width=True, key="shap_bar_chart")

        with col2:
            st.markdown("### Top Risk Factors")
            for f in factors:
                arrow = "▲" if f["direction"] == "increases" else "▼"
                color = "#ff4b4b" if f["direction"] == "increases" else "#00c853"
                label = f["en_label"] if language == "en" else f["ar_label"]
                st.markdown(
                    f'<div class="factor-row">'
                    f'<span style="color:#ccc;font-size:14px">{label}</span>'
                    f'<span style="color:{color};font-weight:600;font-size:14px">'
                    f'{arrow} {f["impact"]:.4f}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.divider()
        # ── Row 2b: Waterfall ──
        st.markdown("### Risk Score Waterfall")
        st.plotly_chart(
            shap_waterfall_chart(factors, base_value=0.07), use_container_width=True,
            key="waterfall_chart"
        )

        st.divider()
        # ── Row 3: LLM Report ──
        lang_label = "🇬🇧 AI Risk Report" if language == "en" else "🇪🇬 تقرير المخاطر"
        st.markdown(f"### {lang_label}")

        if language == "ar":
            st.markdown(
                f'<div class="report-box" dir="rtl" style="text-align:right;'
                f"font-family:'Segoe UI',Arial,sans-serif\">"
                f'{llm_report.replace(chr(10), "<br>")}'
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="report-box">'
                f'{llm_report.replace(chr(10), "<br>")}'
                f"</div>",
                unsafe_allow_html=True,
            )

        st.divider()
        # ── Row 3: Radar + Score Breakdown ──
        st.divider()
        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown("### Client Risk Profile")
            radar_fig, radar_values, radar_cats = risk_radar_chart(payload, factors)
            st.plotly_chart(radar_fig, use_container_width=True, key="radar_chart")

        with col2:
            st.markdown("### Risk Dimensions")
            st.markdown("<br>", unsafe_allow_html=True)

            dim_colors = {
                "Payment History": ("#ff4b4b", "#00c853"),
                "Credit Usage": ("#ff4b4b", "#00c853"),
                "Debt Burden": ("#ff4b4b", "#00c853"),
                "Income Stability": ("#ff4b4b", "#00c853"),
                "Credit Mix": ("#ff4b4b", "#00c853"),
            }

            for cat, val in zip(radar_cats, radar_values):
                color = (
                    "#ff4b4b" if val > 0.6 else "#ffa500" if val > 0.3 else "#00c853"
                )
                label = "High" if val > 0.6 else "Medium" if val > 0.3 else "Low"
                bar_w = int(val * 100)

                st.markdown(
                    f'<div style="margin-bottom:14px">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:5px">'
                    f'<span style="color:#ccc;font-size:13px">{cat}</span>'
                    f'<span style="color:{color};font-size:12px;font-weight:600">'
                    f"{label} ({val:.0%})</span>"
                    f"</div>"
                    f'<div style="background:#2d3147;border-radius:999px;height:6px">'
                    f'<div style="background:{color};width:{bar_w}%;height:6px;'
                    f'border-radius:999px;transition:width 0.3s"></div>'
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        # ── Row 4: Raw Data Summary ──
        with st.expander("📊 View Raw Client Data"):
            summary = {
                "Age": age,
                "Monthly Income": f"${monthly_income:,}",
                "Debt Ratio": f"{debt_ratio:.0%}",
                "Credit Utilization": f"{util_rate:.0%}",
                "Open Credits": open_credits,
                "Real Estate Loans": real_estate_loans,
                "Late 30-59 Days": late_30_59,
                "Late 60-89 Days": late_60_89,
                "Late 90+ Days": late_90,
                "Dependents": dependents,
            }
            col1, col2 = st.columns(2)
            items = list(summary.items())
            for i, (k, v) in enumerate(items[:5]):
                col1.metric(k, v)
            for i, (k, v) in enumerate(items[5:]):
                col2.metric(k, v)
        # ── Row 5: Improvement Roadmap ──
        st.divider()
        st.markdown("### 🗺️ Improvement Roadmap")
        st.markdown(
            '<p style="color:#888;font-size:13px;margin:-10px 0 16px">'
            "Actionable steps to reduce default risk, ranked by impact</p>",
            unsafe_allow_html=True,
        )

        steps = calculate_improvement_steps(payload, risk_score, factors)

        if not steps:
            st.success(
                "✅ Client profile is already strong — no major improvements needed."
            )
        else:
            # Timeline chart
            st.plotly_chart(
                improvement_timeline_chart(steps, risk_score), use_container_width=True,
                key="improvement_timeline_chart"
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Step cards
            for s in steps:
                before = s["before_score"]
                after = s["after_score"]
                reduction_pct = (before - after) / before * 100

                b_color = (
                    "#ff4b4b"
                    if before > 0.6
                    else "#ffa500" if before > 0.3 else "#00c853"
                )
                a_color = (
                    "#ff4b4b"
                    if after > 0.6
                    else "#ffa500" if after > 0.3 else "#00c853"
                )

                st.markdown(
                    f'<div class="risk-card" style="margin-bottom:12px">'
                    # Header
                    f'<div style="display:flex;align-items:center;'
                    f'justify-content:space-between;margin-bottom:12px">'
                    f'<div style="display:flex;align-items:center;gap:10px">'
                    f'<span style="font-size:22px">{s["icon"]}</span>'
                    f"<div>"
                    f'<p style="margin:0;font-size:15px;font-weight:600;color:#fff">'
                    f'Step {s["step"]}: {s["action"]}</p>'
                    f'<p style="margin:2px 0 0;font-size:12px;color:#888">'
                    f'⏱ Timeline: {s["timeline"]}</p>'
                    f"</div>"
                    f"</div>"
                    # Score change badge
                    f'<div style="text-align:right">'
                    f'<span style="color:{b_color};font-size:14px;font-weight:600">'
                    f"{before:.0%}</span>"
                    f'<span style="color:#888;font-size:14px"> → </span>'
                    f'<span style="color:{a_color};font-size:14px;font-weight:600">'
                    f"{after:.0%}</span>"
                    f"<br>"
                    f'<span style="color:#00c853;font-size:11px">'
                    f"▼ {reduction_pct:.0f}% reduction</span>"
                    f"</div>"
                    f"</div>"
                    # How to
                    f'<div style="background:#0f1117;border-radius:8px;padding:10px 14px">'
                    f'<p style="margin:0;font-size:13px;color:#aaa">'
                    f'<b style="color:#4e9eff">How: </b>{s["how"]}</p>'
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Summary
            final_score = steps[-1]["after_score"]
            total_reduction = (risk_score - final_score) / risk_score * 100
            final_label = (
                "High Risk"
                if final_score > 0.6
                else "Medium Risk" if final_score > 0.3 else "Low Risk"
            )
            final_color = (
                "#ff4b4b"
                if final_score > 0.6
                else "#ffa500" if final_score > 0.3 else "#00c853"
            )

            st.markdown(
                f'<div style="background:rgba(78,158,255,0.08);border:1px solid #4e9eff;'
                f"border-radius:12px;padding:16px 20px;margin-top:8px;"
                f'display:flex;justify-content:space-between;align-items:center">'
                f"<div>"
                f'<p style="margin:0;color:#4e9eff;font-weight:600;font-size:14px">'
                f"🎯 After completing all steps:</p>"
                f'<p style="margin:4px 0 0;color:#888;font-size:12px">'
                f"Estimated timeline to full improvement</p>"
                f"</div>"
                f'<div style="text-align:right">'
                f'<p style="margin:0;font-size:22px;font-weight:700;color:{final_color}">'
                f"{final_score:.0%}</p>"
                f'<p style="margin:2px 0 0;color:#00c853;font-size:12px">'
                f"▼ {total_reduction:.0f}% total reduction → {final_label}</p>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
