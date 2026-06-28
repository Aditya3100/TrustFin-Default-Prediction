"""
streamlit_app.py
----------------
TrustFin Bank – Loan Default Risk Dashboard

A fintech-styled Streamlit application that allows risk analysts to:
  • Upload a CSV of customer records
  • Run batch predictions via the LoanRiskPredictor
  • Inspect individual risk scores and factors
  • Review a portfolio-level summary

Run
---
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import pickle

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TrustFin – Loan Risk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS (fintech dark-blue theme)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Background */
    .stApp { background-color: #f0f4f8; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0d1f3c; }
    section[data-testid="stSidebar"] * { color: #d4e3f7 !important; }
    section[data-testid="stSidebar"] .stMarkdown h2 { color: #7eb8f7 !important; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #d0dce8;
        border-radius: 10px;
        padding: 12px 18px;
        box-shadow: 0 2px 6px rgba(0,0,0,.06);
    }

    /* Risk badge colours */
    .badge-high   { background:#fee2e2; color:#b91c1c; padding:4px 10px; border-radius:12px; font-weight:700; }
    .badge-medium { background:#fef3c7; color:#b45309; padding:4px 10px; border-radius:12px; font-weight:700; }
    .badge-low    { background:#d1fae5; color:#065f46; padding:4px 10px; border-radius:12px; font-weight:700; }

    /* Section headers */
    .section-header {
        font-size: 1.1rem; font-weight: 700; color: #0d1f3c;
        border-left: 4px solid #1a73e8; padding-left: 10px;
        margin: 20px 0 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏦 TrustFin Bank")
    st.markdown("**Loan Default Risk Platform**")
    st.markdown("---")
    models_dir = st.text_input("Models directory", value="models")
    threshold  = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.05,
                            help="Probability above which a customer is flagged HIGH risk")
    st.markdown("---")
    st.caption("© 2024 TrustFin Bank – Internal Use Only")


# ---------------------------------------------------------------------------
# Load predictor (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_predictor(models_dir: str, threshold: float):
    try:
        from predict import LoanRiskPredictor
        return LoanRiskPredictor(models_dir=Path(models_dir), threshold=threshold)
    except FileNotFoundError as e:
        return str(e)


predictor = load_predictor(models_dir, threshold)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='color:#0d1f3c;margin-bottom:0'>🏦 TrustFin Loan Default Risk Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#64748b;font-size:1rem'>Upload customer data · Run predictions · Manage portfolio risk</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ---------------------------------------------------------------------------
# Upload section
# ---------------------------------------------------------------------------
st.markdown("<div class='section-header'>📂 Upload Customer Data</div>", unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Upload a CSV file with customer financial data",
    type=["csv"],
    help="Each row should represent one customer. Include financial, credit, and repayment fields.",
)

if uploaded is None:
    st.info("👆 Upload a CSV file to start.  "
            "You can use `data/application_train.csv` (sample rows) for a demo.")
    st.stop()


# ---------------------------------------------------------------------------
# Load and preview data
# ---------------------------------------------------------------------------
df_raw = pd.read_csv(uploaded)
st.success(f"✅ Loaded **{len(df_raw):,} rows** × **{df_raw.shape[1]} columns**")

with st.expander("📋 Preview raw data", expanded=False):
    st.dataframe(df_raw.head(20), use_container_width=True)


# ---------------------------------------------------------------------------
# Run predictions
# ---------------------------------------------------------------------------
if isinstance(predictor, str):
    st.error(f"❌ Could not load model artefacts:\n\n{predictor}")
    st.info("Run `python src/train.py` first to generate model artefacts.")
    st.stop()

with st.spinner("🔍 Running risk predictions …"):
    predictions = predictor.predict_batch(df_raw)

pred_df = pd.DataFrame(predictions)
pred_df["default_probability_pct"] = (pred_df["default_probability"] * 100).round(2)
pred_df["customer_index"] = range(len(pred_df))

# Merge with original ID if present
if "SK_ID_CURR" in df_raw.columns:
    pred_df.insert(0, "SK_ID_CURR", df_raw["SK_ID_CURR"].values)


# ---------------------------------------------------------------------------
# Portfolio summary
# ---------------------------------------------------------------------------
st.markdown("<div class='section-header'>📊 Portfolio Risk Summary</div>", unsafe_allow_html=True)

risk_counts = pred_df["risk_level"].value_counts().to_dict()
high_n   = risk_counts.get("HIGH",   0)
medium_n = risk_counts.get("MEDIUM", 0)
low_n    = risk_counts.get("LOW",    0)
total_n  = len(pred_df)
avg_prob = pred_df["default_probability"].mean() * 100

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Customers",    f"{total_n:,}")
col2.metric("🔴 High Risk",       f"{high_n:,}",   delta=f"{high_n/total_n*100:.1f}%",  delta_color="inverse")
col3.metric("🟡 Medium Risk",     f"{medium_n:,}", delta=f"{medium_n/total_n*100:.1f}%",delta_color="off")
col4.metric("🟢 Low Risk",        f"{low_n:,}",    delta=f"{low_n/total_n*100:.1f}%",   delta_color="normal")
col5.metric("Avg Default Prob",   f"{avg_prob:.2f}%")


# Risk distribution chart
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor("#f0f4f8")

# Bar chart
colors = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#10b981"}
levels = ["HIGH", "MEDIUM", "LOW"]
counts = [risk_counts.get(l, 0) for l in levels]
bars = axes[0].bar(levels, counts, color=[colors[l] for l in levels], edgecolor="white", linewidth=1.5)
axes[0].set_title("Risk Distribution", fontweight="bold", pad=12)
axes[0].set_ylabel("Number of Customers")
axes[0].set_facecolor("#f8fafc")
for bar, cnt in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                  str(cnt), ha="center", va="bottom", fontweight="bold")

# Probability histogram
axes[1].hist(pred_df["default_probability"], bins=40, color="#1a73e8", edgecolor="white",
              linewidth=0.5, alpha=0.9)
axes[1].axvline(threshold, color="#ef4444", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold}")
axes[1].set_title("Default Probability Distribution", fontweight="bold", pad=12)
axes[1].set_xlabel("Default Probability")
axes[1].set_ylabel("Count")
axes[1].legend()
axes[1].set_facecolor("#f8fafc")

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)


# ---------------------------------------------------------------------------
# Full predictions table
# ---------------------------------------------------------------------------
st.markdown("<div class='section-header'>📋 All Predictions</div>", unsafe_allow_html=True)

# Colour-code the risk level column
def colour_risk(val: str) -> str:
    return {
        "HIGH":   "background-color:#fee2e2; color:#b91c1c; font-weight:bold",
        "MEDIUM": "background-color:#fef3c7; color:#b45309; font-weight:bold",
        "LOW":    "background-color:#d1fae5; color:#065f46; font-weight:bold",
    }.get(val, "")

display_cols = [c for c in ["SK_ID_CURR", "risk_level", "default_probability_pct"] if c in pred_df.columns]
styled = (
    pred_df[display_cols]
    .rename(columns={"default_probability_pct": "Default Probability (%)", "risk_level": "Risk Level"})
    .style.applymap(colour_risk, subset=["Risk Level"])
)
st.dataframe(styled, use_container_width=True, height=300)

# Download button
csv_bytes = pred_df.drop(columns=["recommendation", "risk_factors"], errors="ignore").to_csv(index=False).encode()
st.download_button("⬇️  Download predictions CSV", csv_bytes, "predictions.csv", "text/csv")


# ---------------------------------------------------------------------------
# Individual customer deep-dive
# ---------------------------------------------------------------------------
st.markdown("<div class='section-header'>🔎 Individual Customer Deep-Dive</div>", unsafe_allow_html=True)

id_options = pred_df["SK_ID_CURR"].tolist() if "SK_ID_CURR" in pred_df.columns else list(range(len(pred_df)))
selected_id = st.selectbox("Select Customer ID", id_options)

idx = id_options.index(selected_id)
result = predictions[idx]
prob   = result["default_probability"]
level  = result["risk_level"]

badge_class = {"HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}.get(level, "badge-low")

c1, c2 = st.columns([1, 2])
with c1:
    st.markdown(f"**Risk Level:** <span class='{badge_class}'>{level}</span>", unsafe_allow_html=True)
    st.metric("Default Probability", f"{prob*100:.2f}%")

    # Gauge-style probability bar
    bar_color = "#ef4444" if level == "HIGH" else "#f59e0b" if level == "MEDIUM" else "#10b981"
    st.markdown(
        f"""
        <div style="background:#e2e8f0;border-radius:8px;height:14px;margin-top:8px">
          <div style="background:{bar_color};width:{int(prob*100)}%;height:14px;border-radius:8px;"></div>
        </div>
        <p style="font-size:.75rem;color:#64748b;margin-top:4px">{int(prob*100)}% probability of default</p>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown("**⚠️ Risk Factors**")
    for factor in result["risk_factors"]:
        st.markdown(f"- {factor}")

    st.markdown("**💡 Recommendations**")
    for rec in result["recommendation"]:
        st.markdown(f"- {rec}")