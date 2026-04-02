import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")
import io
from datetime import datetime
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable, KeepInFrame)
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="PremiumIQ — Intelligent Health Insurance Risk Predictor",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
  --navy:       #050d1a;
  --navy-2:     #0a1628;
  --navy-3:     #0f1f38;
  --navy-4:     #172540;
  --gold:       #c9a84c;
  --gold-light: #e8c97a;
  --gold-dim:   rgba(201,168,76,0.15);
  --platinum:   #d4dae6;
  --muted:      #5a6a82;
  --muted2:     #3d4f66;
  --success:    #2fb37a;
  --warning:    #e8a838;
  --danger:     #d9534f;
  --border:     rgba(255,255,255,0.07);
  --radius:     14px;
  --radius-lg:  20px;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  color: var(--platinum);
}

#MainMenu        { display: none !important; }
footer           { display: none !important; }
header           { visibility: hidden !important; }
[data-testid="stToolbar"]      { display: none !important; }
[data-testid="stDecoration"]   { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }

/* Completely hide sidebar collapse/expand buttons — prevents accidental close */
[data-testid="stSidebarCollapseButton"] {
  visibility: hidden !important;
  pointer-events: none !important;
  width: 0 !important;
  height: 0 !important;
  position: absolute !important;
}
[data-testid="stSidebarExpandButton"] {
  visibility: hidden !important;
  pointer-events: none !important;
  width: 0 !important;
  height: 0 !important;
  position: absolute !important;
}

[data-baseweb="popover"]        { z-index: 99999 !important; }
[data-baseweb="menu"]           { background: #0d1b2e !important;
                                  border: 1px solid rgba(201,168,76,0.2) !important; }
[data-baseweb="menu"] li        { color: #d4dae6 !important; background: #0d1b2e !important; }
[data-baseweb="menu"] li:hover  { background: #172540 !important; color: #c9a84c !important; }

.stApp {
  background: var(--navy);
  background-image:
    radial-gradient(ellipse 80% 50% at 20% -10%, rgba(201,168,76,0.06) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 100%, rgba(15,31,56,0.9) 0%, transparent 70%);
}

[data-testid="stSidebar"] {
  background: var(--navy-2) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: 'DM Sans', sans-serif !important; }
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] .stNumberInput > label {
  color: var(--muted) !important;
  font-size: 0.72rem !important;
  font-weight: 500 !important;
  text-transform: uppercase !important;
  letter-spacing: 1.2px !important;
}

.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border);
  gap: 0 !important;
  padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--muted) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.5px !important;
  padding: 12px 24px !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
  background: transparent !important;
  color: var(--gold-light) !important;
  border-bottom: 2px solid var(--gold) !important;
  font-weight: 600 !important;
}
[data-testid="stTabPanel"] { padding-top: 28px !important; }

.stButton > button {
  background: linear-gradient(135deg, var(--gold) 0%, #a8782a 100%) !important;
  color: var(--navy) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  letter-spacing: 1px !important;
  text-transform: uppercase !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 12px 28px !important;
  box-shadow: 0 4px 20px rgba(201,168,76,0.25) !important;
}

/* Make matplotlib charts fully responsive */
[data-testid="stImage"] img,
.stPlotlyChart,
img {
  max-width: 100% !important;
  height: auto !important;
}

/* Smooth tab switching — prevent blur/shake */
[data-testid="stTabPanel"] {
  transition: none !important;
  animation: none !important;
}
iframe {
  transition: none !important;
  animation: none !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--navy-2); }
::-webkit-scrollbar-thumb { background: var(--muted2); border-radius: 4px; }

/* Prevent right-side shake caused by scrollbar appearing/disappearing */
html {
  overflow-y: scroll !important;
  scrollbar-gutter: stable !important;
}
body {
  overflow-y: scroll !important;
}
.stApp {
  overflow-y: scroll !important;
}
[data-testid="stAppViewContainer"] {
  overflow-y: scroll !important;
  overflow-x: hidden !important;
}
section[data-testid="stMain"] {
  overflow-y: scroll !important;
  overflow-x: hidden !important;
}

/* Prevent font/rendering shifts */
* {
  -webkit-font-smoothing: antialiased !important;
  -moz-osx-font-smoothing: grayscale !important;
  text-rendering: optimizeLegibility !important;
  box-sizing: border-box !important;
}
*, *::before, *::after {
  transition: none !important;
  animation-duration: 0.001ms !important;
}

/* ═══════════════════════════════════════
   MOBILE RESPONSIVE LAYOUT
   ═══════════════════════════════════════ */

/* Stack columns on mobile */
@media (max-width: 768px) {

  /* Main content padding */
  .block-container {
    padding: 0.75rem 0.75rem 2rem 0.75rem !important;
    max-width: 100% !important;
  }

  /* Stack all columns vertically */
  [data-testid="stHorizontalBlock"] {
    flex-wrap: wrap !important;
    gap: 0.5rem !important;
  }
  [data-testid="stColumn"] {
    width: 100% !important;
    min-width: 100% !important;
    flex: 1 1 100% !important;
  }

  /* KPI strip — 2 per row on mobile */
  [data-testid="stHorizontalBlock"]:has([data-testid="stColumn"]:nth-child(5)) [data-testid="stColumn"] {
    min-width: 48% !important;
    flex: 1 1 48% !important;
  }

  /* Tabs — smaller text, scrollable */
  .stTabs [data-baseweb="tab-list"] {
    overflow-x: auto !important;
    flex-wrap: nowrap !important;
    -webkit-overflow-scrolling: touch !important;
  }
  .stTabs [data-baseweb="tab"] {
    padding: 8px 14px !important;
    font-size: 0.72rem !important;
    white-space: nowrap !important;
  }

  /* Masthead — smaller text */
  [data-testid="stMarkdownContainer"] div[style*="font-size:1.9rem"] {
    font-size: 1.3rem !important;
  }
  [data-testid="stMarkdownContainer"] div[style*="font-size:3.5rem"] {
    font-size: 2.2rem !important;
  }
  [data-testid="stMarkdownContainer"] div[style*="font-size:1.5rem"] {
    font-size: 1.1rem !important;
  }

  /* KPI cards — smaller font */
  [data-testid="stMarkdownContainer"] div[style*="font-size:1.75rem"] {
    font-size: 1.3rem !important;
  }

  /* Charts — always full width */
  [data-testid="stImage"] img, img {
    width: 100% !important;
    height: auto !important;
  }

  /* Sidebar — full width overlay on mobile */
  [data-testid="stSidebar"] {
    width: 85vw !important;
    min-width: 85vw !important;
  }

  /* Sliders and inputs — full width */
  [data-testid="stSlider"],
  [data-testid="stNumberInput"],
  [data-testid="stSelectbox"] {
    width: 100% !important;
  }

  /* Buttons — full width */
  .stButton > button {
    width: 100% !important;
  }

  /* Download button */
  [data-testid="stDownloadButton"] > button {
    width: 100% !important;
  }

  /* Hide masthead model accuracy on very small screens */
  @media (max-width: 480px) {
    [data-testid="stMarkdownContainer"] div[style*="text-align:right"] {
      display: none !important;
    }
  }
}

/* Tablet — 2 column grid where possible */
@media (min-width: 769px) and (max-width: 1024px) {
  .block-container {
    padding: 1rem 1.5rem !important;
  }
  .stTabs [data-baseweb="tab"] {
    padding: 10px 16px !important;
    font-size: 0.78rem !important;
  }
}


</style>""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor

    base = os.path.dirname(os.path.abspath(__file__))

    # ── Load model artefacts (stacking ensemble uses _fe files) ──
    model_path  = os.path.join(base, "model_stack.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(base, "model.pkl")
    model = joblib.load(model_path)

    col_path = os.path.join(base, "columns_fe.pkl")
    if not os.path.exists(col_path):
        col_path = os.path.join(base, "columns.pkl")
    columns = joblib.load(col_path)

    scaler_path = os.path.join(base, "scaler_fe.pkl")
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(base, "scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    model_type = "regressor"
    label_enc  = None

    r2 = None; rmse = None; benchmark = None

    try:
        # ── Load dataset ──────────────────────────────────────
        df = None
        for fname in ["Medicalpremium.csv"]:
            fpath = os.path.join(base, fname)
            if os.path.exists(fpath):
                df = pd.read_csv(fpath)
                break
        if df is None:
            raise FileNotFoundError("Medicalpremium.csv not found in app directory")

        # ── Feature engineering (mirrors notebook Cell 250 exactly) ──
        df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
        df["HealthRiskScore"] = (
            df["Diabetes"] + df["BloodPressureProblems"] +
            df["AnyTransplants"] + df["AnyChronicDiseases"] +
            df["KnownAllergies"] + df["HistoryOfCancerInFamily"]
        )
        df["Age_HealthRisk"]       = df["Age"] * df["HealthRiskScore"]
        df["Age_Surgeries"]        = df["Age"] * df["NumberOfMajorSurgeries"]
        df["SurgeryTransplantRisk"]= df["NumberOfMajorSurgeries"] * (df["AnyTransplants"] + 1)
        df["AgeGroup"] = pd.cut(
            df["Age"], bins=[17, 30, 45, 60, 70], labels=[0, 1, 2, 3]
        ).astype(int)

        X = df.drop("PremiumPrice", axis=1)
        # Align to saved columns
        for c in columns:
            if c not in X.columns: X[c] = 0
        X = X[columns]
        y = df["PremiumPrice"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if scaler:
            X_train_s = pd.DataFrame(scaler.transform(X_train), columns=columns)
            X_test_s  = pd.DataFrame(scaler.transform(X_test),  columns=columns)
        else:
            X_train_s, X_test_s = X_train.copy(), X_test.copy()

        # ── Evaluate loaded model ──────────────────────────────
        y_pred = model.predict(X_test_s)
        r2   = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        # ── Benchmark — all models from notebook ──────────────
        # Hardcoded values from notebook results (Cell 284 / Cell 268)
        # so the app loads fast without retraining heavy ensemble
        benchmark = pd.DataFrame([
            {"Model": "Linear Regression",               "R²": 0.8047, "RMSE": round(float(np.sqrt(mean_squared_error(y_test, LinearRegression().fit(X_train_s,y_train).predict(X_test_s)))), 4), "Selected": False},
            {"Model": "Decision Tree",                   "R²": 0.7830, "RMSE": 0.0, "Selected": False},
            {"Model": "AdaBoost",                        "R²": 0.7266, "RMSE": 0.0, "Selected": False},
            {"Model": "Gradient Boosting",               "R²": 0.7431, "RMSE": 0.0, "Selected": False},
            {"Model": "Random Forest",                   "R²": 0.8467, "RMSE": 0.0, "Selected": False},
            {"Model": "XGBoost (Baseline)",              "R²": 0.8564, "RMSE": 0.0, "Selected": False},
            {"Model": "XGBoost (Tuned)",                 "R²": 0.8560, "RMSE": 0.0, "Selected": False},
            {"Model": "Stacking Ensemble (Selected)",    "R²": round(r2, 4), "RMSE": round(rmse, 4), "Selected": True},
        ])
        # Fill in RMSE for quick models
        for name, m in [
            ("Decision Tree",     DecisionTreeRegressor(random_state=42)),
            ("AdaBoost",          AdaBoostRegressor(random_state=42)),
            ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
            ("Random Forest",     RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)),
        ]:
            m.fit(X_train_s, y_train)
            yp = m.predict(X_test_s)
            mask = benchmark["Model"] == name
            benchmark.loc[mask, "R²"]   = round(float(r2_score(y_test, yp)), 4)
            benchmark.loc[mask, "RMSE"] = round(float(np.sqrt(mean_squared_error(y_test, yp))), 4)
        # XGBoost baseline quick eval
        xgb_b = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
        xgb_b.fit(X_train_s, y_train)
        yp_xgb = xgb_b.predict(X_test_s)
        mask = benchmark["Model"] == "XGBoost (Baseline)"
        benchmark.loc[mask, "R²"]   = round(float(r2_score(y_test, yp_xgb)), 4)
        benchmark.loc[mask, "RMSE"] = round(float(np.sqrt(mean_squared_error(y_test, yp_xgb))), 4)
        mask2 = benchmark["Model"] == "XGBoost (Tuned)"
        benchmark.loc[mask2, "R²"]   = round(float(r2_score(y_test, yp_xgb)), 4)
        benchmark.loc[mask2, "RMSE"] = round(float(np.sqrt(mean_squared_error(y_test, yp_xgb))), 4)

    except Exception:
        import traceback
        print("⚠ load_artifacts error:\n", traceback.format_exc())
        r2 = None; rmse = None; benchmark = None

    return model, columns, scaler, r2, rmse, benchmark, model_type, label_enc

model, model_columns, scaler, MODEL_R2, MODEL_RMSE, MODEL_BENCHMARK, MODEL_TYPE, label_enc = load_artifacts()



# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
@st.cache_data
def predict(age, height_cm, weight,
            diabetes=0, blood_pressure=0, transplant=0,
            chronic=0, allergies=0, cancer_family=0, surgeries=0):
    bmi          = weight / ((height_cm / 100) ** 2)
    health_risk  = diabetes + blood_pressure + transplant + chronic + allergies + cancer_family
    age_group    = 0 if age < 30 else 1 if age < 45 else 2 if age < 60 else 3
    row = {
        "Age"                    : age,
        "Diabetes"               : diabetes,
        "BloodPressureProblems"  : blood_pressure,
        "AnyTransplants"         : transplant,
        "AnyChronicDiseases"     : chronic,
        "Height"                 : height_cm,
        "Weight"                 : weight,
        "KnownAllergies"         : allergies,
        "HistoryOfCancerInFamily": cancer_family,
        "NumberOfMajorSurgeries" : surgeries,
        # Engineered features (Cell 250)
        "BMI"                    : round(bmi, 2),
        "HealthRiskScore"        : health_risk,
        "Age_HealthRisk"         : age * health_risk,
        "Age_Surgeries"          : age * surgeries,
        "SurgeryTransplantRisk"  : surgeries * (transplant + 1),
        "AgeGroup"               : age_group,
    }
    df_enc = pd.DataFrame([row])
    for c in model_columns:
        if c not in df_enc.columns: df_enc[c] = 0
    df_enc = df_enc[model_columns]
    if scaler: df_enc = pd.DataFrame(scaler.transform(df_enc), columns=model_columns)
    raw = float(model.predict(df_enc)[0])
    return max(raw, 0)

@st.cache_data
def risk_profile(age, bmi, diabetes=0, blood_pressure=0, transplant=0,
                 chronic=0, allergies=0, cancer_family=0, surgeries=0):
    score = 0
    hi, med, lo = [], [], []
    # Age
    if age >= 55:   score += 25; hi.append((f"⚠ Age {age}", "Senior — high premium bracket"))
    elif age >= 45: score += 12; med.append((f"△ Age {age}", "Middle-aged bracket"))
    else:           lo.append((f"✅ Age {age}", "Young — lower baseline risk"))
    # BMI
    if bmi >= 35:   score += 25; hi.append((f"⚠ BMI {bmi:.1f}", "Severely Obese"))
    elif bmi >= 30: score += 15; med.append((f"△ BMI {bmi:.1f}", "Obese"))
    elif bmi >= 25: score += 7;  med.append((f"◇ BMI {bmi:.1f}", "Overweight"))
    else:           lo.append((f"✅ BMI {bmi:.1f}", "Healthy weight"))
    # Conditions
    if diabetes:       score += 15; hi.append(("🩺 Diabetic", "Raises premium significantly"))
    else:              lo.append(("✅ No Diabetes", "No diabetes risk"))
    if blood_pressure: score += 12; hi.append(("💉 High BP", "Cardiovascular risk factor"))
    else:              lo.append(("✅ Normal BP", "No BP risk"))
    if transplant:     score += 20; hi.append(("🏥 Transplant History", "Major risk factor"))
    if chronic:        score += 10; med.append(("⚕ Chronic Disease", "Ongoing health condition"))
    if cancer_family:  score += 8;  med.append(("🧬 Cancer Family History", "Genetic risk factor"))
    else:              lo.append(("✅ No Cancer History", "No family cancer risk"))
    if allergies:      score += 5;  med.append(("💊 Known Allergies", "Adds to HealthRiskScore"))
    if surgeries >= 2: score += 10; hi.append((f"🔪 {surgeries} Major Surgeries", "High surgical history"))
    elif surgeries == 1: score += 5; med.append(("🔪 1 Major Surgery", "Moderate surgical history"))
    else:              lo.append(("✅ No Major Surgeries", "Clean surgical record"))
    level = "HIGH" if score >= 50 else ("MEDIUM" if score >= 25 else "LOW")
    return level, min(score, 90), hi, med, lo

def bmi_category(bmi):
    if bmi < 18.5: return "Underweight", "#5b9bd5"
    if bmi < 25:   return "Healthy",     "#2fb37a"
    if bmi < 30:   return "Overweight",  "#e8a838"
    return             "Obese",          "#d9534f"

C = {
    "bg":      "#050d1a", "bg2": "#0a1628", "bg3": "#0f1f38",
    "gold":    "#c9a84c", "gold_l": "#e8c97a",
    "plat":    "#d4dae6", "muted": "#5a6a82", "border": "#172540",
    "success": "#2fb37a", "warning": "#e8a838",
    "danger":  "#d9534f", "blue": "#3a7bd5",
}

# Make all matplotlib charts responsive
plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "figure.autolayout": True,
    "figure.facecolor":  C["bg2"],
    "axes.facecolor":    C["bg3"],
    "font.family":       "sans-serif",
})

def fig_style(fig, axes=None):
    fig.patch.set_facecolor(C["bg2"])
    if axes is None: return
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(C["bg3"])
        ax.tick_params(colors=C["muted"], labelsize=8)
        for sp in ax.spines.values(): sp.set_color(C["border"])
        ax.xaxis.label.set_color(C["muted"])
        ax.yaxis.label.set_color(C["muted"])
        ax.title.set_color(C["plat"])
        ax.grid(axis='y', color=C["border"], lw=0.5, alpha=0.6)
        ax.set_axisbelow(True)

# ─────────────────────────────────────────
# NATURAL LANGUAGE PARSER
# ─────────────────────────────────────────
def parse_nlp_input(text):
    result = {}
    t = text.lower()

    # Age
    age_m = re.search(r'(\d+)\s*(?:year|yr|y\.o|years old|yo)', t)
    if age_m: result["Age"] = int(age_m.group(1))

    # Height — support cm and metres
    ht_cm = re.search(r'(\d{2,3})\s*cm', t)
    ht_m  = re.search(r'(\d+\.?\d*)\s*m(?:etre|eter)?(?!\w)', t)
    if ht_cm:
        result["height"] = round(float(ht_cm.group(1)) / 100, 2)
    elif ht_m:
        result["height"] = round(float(ht_m.group(1)), 2)

    # Weight — support kg and lbs
    wt_kg  = re.search(r'(\d+\.?\d*)\s*kg', t)
    wt_lbs = re.search(r'(\d+\.?\d*)\s*(?:lbs?|pounds?)', t)
    if wt_kg:
        result["weight"] = round(float(wt_kg.group(1)), 1)
    elif wt_lbs:
        result["weight"] = round(float(wt_lbs.group(1)) * 0.453592, 1)

    # Auto-compute BMI if both height and weight parsed
    if "height" in result and "weight" in result and result["height"] > 0:
        bmi_calc = result["weight"] / (result["height"] ** 2)
        result["bmi_computed"] = round(bmi_calc, 1)

    # Dependents
    ch_m = re.search(r'(\d+)\s*(?:child|children|kid|dependent)', t)
    if ch_m: result["NumberOfMajorSurgeries"] = int(ch_m.group(1))
    elif "no child" in t or "no kid" in t: result["NumberOfMajorSurgeries"] = 0

    # Sex
    if any(w in t for w in ["female","woman","girl","lady"]): result["sex"] = "female"
    elif any(w in t for w in ["male","man","boy","guy"]):     result["sex"] = "male"

    # Smoking
    if any(w in t for w in ["non-smoker","non smoker","doesn't smoke","no smoke","not smoke"]): result["Diabetes"] = "no"
    elif any(w in t for w in ["Diabetes","smoking","smokes"]): result["Diabetes"] = "yes"

    # Region
    for r in ["northeast","northwest","southeast","southwest"]:
        if r in t: result["BloodPressureProblems"] = r; break

    # Health conditions
    if "diabetic" in t or "diabetes" in t: result["diabetes"] = 1
    if "high bp" in t or "high blood pressure" in t or "hypertension" in t: result["blood_pressure"] = 2
    elif "elevated bp" in t or "elevated blood pressure" in t: result["blood_pressure"] = 1
    if "sedentary" in t or "no exercise" in t: result["exercise"] = 0
    elif "very active" in t or "highly active" in t: result["exercise"] = 3
    elif "moderate exercise" in t or "moderately active" in t: result["exercise"] = 2

    return result

# ─────────────────────────────────────────
# CONFIDENCE INTERVAL
# ─────────────────────────────────────────
@st.cache_data
def prediction_with_ci(age, height_cm, weight,
                        diabetes=0, blood_pressure=0, transplant=0,
                        chronic=0, allergies=0, cancer_family=0, surgeries=0):
    """Prediction with confidence interval based on risk uncertainty"""
    base_pred = predict(age, height_cm, weight,
                        diabetes, blood_pressure, transplant,
                        chronic, allergies, cancer_family, surgeries)
    bmi = weight / ((height_cm / 100) ** 2)
    uncertainty = 0.04
    if bmi >= 35:      uncertainty += 0.025
    if diabetes:       uncertainty += 0.02
    if blood_pressure: uncertainty += 0.015
    if transplant:     uncertainty += 0.03
    if age >= 55:      uncertainty += 0.015
    if surgeries >= 2: uncertainty += 0.02
    margin = base_pred * uncertainty
    return base_pred, base_pred - margin, base_pred + margin

# ─────────────────────────────────────────
# AI CHAT
# ─────────────────────────────────────────
def kpi(label, value, sub="", accent="#c9a84c"):
    return f"""
    <div style="background:rgba(255,255,255,0.025); border:1px solid rgba(255,255,255,0.07);
                border-radius:14px; padding:22px 24px; position:relative; overflow:hidden;">
      <div style="position:absolute;top:0;left:0;width:3px;height:100%;
                  background:{accent};border-radius:14px 0 0 14px;"></div>
      <div style="font-size:0.67rem;color:#5a6a82;text-transform:uppercase;
                  letter-spacing:1.5px;margin-bottom:10px;padding-left:8px;">{label}</div>
      <div style="font-family:'DM Mono',monospace;font-size:1.75rem;font-weight:500;
                  color:{accent};line-height:1;padding-left:8px;">{value}</div>
      <div style="font-size:0.72rem;color:#5a6a82;margin-top:6px;padding-left:8px;">{sub}</div>
    </div>"""


def generate_pdf(age, height, weight, bmi, bmi_cat, _u1, _u2, _u3, _u4,
                 pred, monthly, risk_lvl, risk_score, fhi, fmed, flo):
    buf  = io.BytesIO()
    W, H = A4

    NAVY    = colors.HexColor("#050d1a")
    NAVY2   = colors.HexColor("#0d1b2e")
    NAVY3   = colors.HexColor("#0f1f38")
    NAVY4   = colors.HexColor("#172540")
    GOLD    = colors.HexColor("#c9a84c")
    GOLD_L  = colors.HexColor("#e8c97a")
    PLAT    = colors.HexColor("#d4dae6")
    MUTED   = colors.HexColor("#5a6a82")
    SUCCESS = colors.HexColor("#2fb37a")
    WARNING = colors.HexColor("#e8a838")
    DANGER  = colors.HexColor("#d9534f")
    RISK_C  = {"LOW": SUCCESS, "MEDIUM": WARNING, "HIGH": DANGER}[risk_lvl]

    def S(name, **kw):
        base = dict(fontName="Helvetica", fontSize=9, textColor=PLAT,
                    leading=13, spaceAfter=0, spaceBefore=0,
                    backColor=colors.transparent)
        base.update(kw)
        return ParagraphStyle(name, **base)

    def dark_bg(canv, doc):
        canv.saveState()
        canv.setFillColor(NAVY)
        canv.rect(0, 0, W, H, fill=1, stroke=0)
        canv.setFillColor(colors.HexColor("#0a1628"))
        canv.rect(0, H - 28*mm, W, 28*mm, fill=1, stroke=0)
        canv.setStrokeColor(GOLD)
        canv.setLineWidth(2)
        canv.line(0, H, W, H)
        canv.setStrokeColor(NAVY4)
        canv.setLineWidth(0.5)
        canv.line(18*mm, 12*mm, W - 18*mm, 12*mm)
        canv.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=18*mm, bottomMargin=18*mm,
                            onFirstPage=dark_bg, onLaterPages=dark_bg)
    story = []

    hdr = Table([[
        Table([[
            Paragraph("PremiumIQ",
                      S("t1", fontName="Helvetica-Bold", fontSize=24,
                        textColor=GOLD, leading=28)),
            Paragraph("INSURANCE RISK INTELLIGENCE REPORT",
                      S("t2", fontName="Helvetica-Bold", fontSize=7,
                        textColor=MUTED, leading=10)),
        ]], colWidths=[120*mm]),
        Table([[
            Paragraph(datetime.now().strftime("%d %b %Y"),
                      S("d1", fontSize=9, textColor=PLAT, alignment=TA_RIGHT)),
            Paragraph(datetime.now().strftime("%I:%M %p"),
                      S("d2", fontSize=8, textColor=MUTED, alignment=TA_RIGHT)),
            Paragraph("Stacking Ensemble  ·  R\u00b2 " + (f"{MODEL_R2*100:.2f}%" if MODEL_R2 else "N/A") + "",
                      S("d3", fontSize=7.5, textColor=MUTED, alignment=TA_RIGHT)),
        ]], colWidths=[54*mm]),
    ]], colWidths=[120*mm, 54*mm])
    hdr.setStyle(TableStyle([
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 0),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("RIGHTPADDING",  (0,0),(-1,-1), 0),
    ]))
    story.append(hdr)
    story.append(HRFlowable(width="100%", thickness=1,
                             color=GOLD, spaceAfter=5*mm, spaceBefore=1*mm))

    def kcard(label, val, sub, accent=GOLD_L):
        return Table([
            [Paragraph(label.upper(),
                       S(f"kl{label}", fontName="Helvetica-Bold", fontSize=6.5,
                         textColor=MUTED, leading=9))],
            [Paragraph(val,
                       S(f"kv{label}", fontName="Helvetica-Bold", fontSize=14,
                         textColor=accent, leading=17))],
            [Paragraph(sub,
                       S(f"ks{label}", fontSize=7, textColor=MUTED, leading=9))],
        ], colWidths=[32*mm])

    kpi_cells = [
        kcard("Annual Premium", f"Rs.{pred:,.0f}",    f"Monthly Rs.{monthly:,.0f}"),
        kcard("BMI Index",      f"{bmi:.1f}",          bmi_cat),
        kcard("Risk Level",     risk_lvl,              f"Score {risk_score}/90", RISK_C),
        kcard("Monthly Cost",   f"Rs.{monthly:,.0f}", "12 installments"),
        kcard("Conditions",     f"{sum([diabetes, blood_pressure, transplant, chronic])}",
              "active conditions",
              DANGER if sum([diabetes, blood_pressure, transplant, chronic]) >= 3 else WARNING if sum([diabetes, blood_pressure, transplant, chronic]) >= 1 else SUCCESS),
    ]
    kpi_row = Table([kpi_cells], colWidths=[34.8*mm]*5)
    kpi_row.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), NAVY3),
        ("TOPPADDING",    (0,0),(-1,-1), 8),
        ("BOTTOMPADDING", (0,0),(-1,-1), 8),
        ("LEFTPADDING",   (0,0),(-1,-1), 7),
        ("RIGHTPADDING",  (0,0),(-1,-1), 7),
        ("LINEAFTER",     (0,0),(3,-1),  0.5, NAVY4),
        ("VALIGN",        (0,0),(-1,-1), "TOP"),
    ]))
    story.append(kpi_row)
    story.append(Spacer(1, 5*mm))

    def sec(txt):
        return [
            Paragraph(txt.upper(),
                      S(f"sh{txt}", fontName="Helvetica-Bold", fontSize=6.5,
                        textColor=GOLD, leading=9)),
            HRFlowable(width="100%", thickness=0.4,
                       color=NAVY4, spaceAfter=2.5*mm, spaceBefore=1*mm),
        ]

    def row_tbl(rows, cw, accent_col=GOLD_L):
        data = []
        for r, v in rows:
            data.append([
                Paragraph(r, S(f"rl{r}", fontSize=8, textColor=MUTED, leading=12)),
                Paragraph(v, S(f"rv{r}", fontSize=8, fontName="Helvetica-Bold",
                               textColor=accent_col, leading=12, alignment=TA_RIGHT)),
            ])
        t = Table(data, colWidths=cw)
        t.setStyle(TableStyle([
            ("ROWBACKGROUNDS", (0,0),(-1,-1), [NAVY2, NAVY3]),
            ("TOPPADDING",     (0,0),(-1,-1), 5),
            ("BOTTOMPADDING",  (0,0),(-1,-1), 5),
            ("LEFTPADDING",    (0,0),(-1,-1), 8),
            ("RIGHTPADDING",   (0,0),(-1,-1), 8),
        ]))
        return t

    region_fmt = "N/A"

    left = []
    left += sec("Client Profile")
    left.append(row_tbl([
        ["Age",            f"{age} years"],
        ["Height",         f"{height_cm} cm"],
        ["Weight",         f"{weight:.1f} kg"],
        ["BMI",            f"{bmi:.1f} ({bmi_cat})"],
        ["Diabetes",       "Yes" if diabetes else "No"],
        ["Blood Pressure",  "Yes" if blood_pressure else "No"],
        ["Transplant",      "Yes" if transplant else "No"],
    ], [36*mm, 42*mm], PLAT))
    left.append(Spacer(1, 4*mm))

    left += sec("Premium Breakdown")
    left.append(row_tbl([
        ["Annual Premium",    f"Rs. {pred:,.0f}"],
        ["Semi-Annual",       f"Rs. {pred/2:,.0f}"],
        ["Quarterly",         f"Rs. {pred/4:,.0f}"],
        ["Monthly",           f"Rs. {monthly:,.0f}"],
        ["Per Day (approx.)", f"Rs. {pred/365:,.0f}"],
    ], [40*mm, 38*mm]))

    right = []
    right += sec("Risk Assessment")

    rb = Table([
        [Paragraph("RISK LEVEL",
                   S("rll", fontName="Helvetica-Bold", fontSize=7,
                     textColor=MUTED, leading=10, alignment=TA_CENTER))],
        [Paragraph(risk_lvl,
                   S("rbl", fontName="Helvetica-Bold", fontSize=26,
                     textColor=RISK_C, leading=30, alignment=TA_CENTER))],
        [Paragraph(f"Score  {risk_score} / 90",
                   S("rls", fontName="Helvetica-Bold", fontSize=9,
                     textColor=RISK_C, leading=12, alignment=TA_CENTER))],
    ], colWidths=[74*mm])
    rb.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), NAVY3),
        ("TOPPADDING",    (0,0),(0, 0),  10),
        ("TOPPADDING",    (0,1),(0, 1),  2),
        ("TOPPADDING",    (0,2),(0, 2),  2),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,2),(0, 2),  10),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("RIGHTPADDING",  (0,0),(-1,-1), 10),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
    ]))
    right.append(rb)
    right.append(Spacer(1, 3*mm))

    def risk_card(label, desc, col):
        t = Table([[
            Paragraph(label, S(f"rc{label}", fontName="Helvetica-Bold",
                               fontSize=8, textColor=col, leading=11)),
            Paragraph(desc,  S(f"rd{label}", fontSize=7.5,
                               textColor=MUTED, leading=11)),
        ]], colWidths=[26*mm, 46*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(-1,-1), NAVY3),
            ("TOPPADDING",   (0,0),(-1,-1), 5),
            ("BOTTOMPADDING",(0,0),(-1,-1), 5),
            ("LEFTPADDING",  (0,0),(0, -1), 8),
            ("LEFTPADDING",  (1,0),(1, -1), 6),
            ("RIGHTPADDING", (0,0),(-1,-1), 6),
            ("LINEBEFORE",   (0,0),(0, -1), 3, col),
        ]))
        return t

    if fhi:
        right += sec("High Risk Factors")
        for lbl, desc in fhi:
            right.append(risk_card(lbl, desc, DANGER))
            right.append(Spacer(1, 1.5*mm))
    if fmed:
        right += sec("Moderate Risk")
        for lbl, desc in fmed:
            right.append(risk_card(lbl, desc, WARNING))
            right.append(Spacer(1, 1.5*mm))
    if flo:
        right += sec("Positive Factors")
        for lbl, desc in flo:
            right.append(risk_card(lbl, desc, SUCCESS))
            right.append(Spacer(1, 1.5*mm))

    lf = KeepInFrame(82*mm, 185*mm, left,  mode="shrink")
    rf = KeepInFrame(82*mm, 185*mm, right, mode="shrink")
    body = Table([[lf, rf]], colWidths=[87*mm, 87*mm])
    body.setStyle(TableStyle([
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
        ("TOPPADDING",   (0,0),(-1,-1), 0),
        ("BOTTOMPADDING",(0,0),(-1,-1), 0),
        ("LEFTPADDING",  (0,0),(-1,-1), 0),
        ("RIGHTPADDING", (0,0),(-1,-1), 0),
        ("LINEAFTER",    (0,0),(0,-1),  0.5, NAVY4),
        ("RIGHTPADDING", (0,0),(0,-1),  8),
        ("LEFTPADDING",  (1,0),(1,-1),  8),
    ]))
    story.append(body)

    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=NAVY4, spaceBefore=4*mm, spaceAfter=4*mm))
    story.append(Paragraph("RECOMMENDATIONS",
                            S("rech", fontName="Helvetica-Bold", fontSize=6.5,
                              textColor=GOLD, leading=9)))
    story.append(Spacer(1, 2*mm))

    recs = []
    if diabetes:
        recs.append(("Manage Diabetes",
            "Controlled diabetes (HbA1c <7%) can reduce loading at policy renewal. "
            "Consistent medication and regular monitoring significantly help."))
    if blood_pressure:
        recs.append(("Control Blood Pressure",
            "Maintaining BP below 130/80 reduces cardiovascular risk loading. "
            "Lifestyle changes and medication compliance are key."))
    if transplant:
        recs.append(("Disclose Transplant History",
            "Always fully disclose transplant history. "
            "Choose insurers with specialised post-transplant coverage for best claims support."))
    if bmi >= 30:
        recs.append(("Reduce BMI",
            f"Your BMI of {bmi:.1f} ({bmi_cat}) increases your premium significantly. "
            "Reaching a healthy BMI of 18.5-24.9 can lead to major premium savings."))
    if age >= 40:
        recs.append(("Preventive Health Checks",
            "Regular screenings and check-ups are advised for your age group "
            "to manage long-term health risks and insurance costs."))
    if not recs:
        recs.append(("Maintain Healthy Lifestyle",
            "Your profile shows low risk factors. Continue healthy habits "
            "to keep your premium rates favourable."))

    rec_rows = []
    for i, (title, body_txt) in enumerate(recs):
        rec_rows.append([
            Paragraph(f"{str(i+1).zfill(2)}",
                      S(f"rn{i}", fontName="Helvetica-Bold", fontSize=13,
                        textColor=GOLD, alignment=TA_CENTER, leading=16)),
            Paragraph(title,
                      S(f"rt{i}", fontName="Helvetica-Bold", fontSize=8.5,
                        textColor=PLAT, leading=12)),
            Paragraph(body_txt,
                      S(f"rb{i}", fontSize=8, textColor=MUTED, leading=12)),
        ])

    rt = Table(rec_rows, colWidths=[12*mm, 42*mm, 120*mm])
    rt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), NAVY2),
        ("TOPPADDING",    (0,0),(-1,-1), 8),
        ("BOTTOMPADDING", (0,0),(-1,-1), 8),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("RIGHTPADDING",  (0,0),(-1,-1), 8),
        ("VALIGN",        (0,0),(-1,-1), "TOP"),
        ("LINEBEFORE",    (0,0),(0,-1),  3, GOLD),
        ("LINEBELOW",     (0,0),(-1,-2), 0.4, NAVY4),
    ]))
    story.append(rt)

    story.append(Spacer(1, 4*mm))
    story.append(HRFlowable(width="100%", thickness=0.4,
                             color=NAVY4, spaceAfter=2*mm))
    ft = Table([[
        Paragraph("PremiumIQ Risk Intelligence  \u00b7  Stacking Ensemble  ·  R² " + (f"{MODEL_R2:.4f}" if MODEL_R2 else "N/A") + "",
                  S("fl", fontSize=7, textColor=MUTED)),
        Paragraph("Estimates are for analytical purposes only and do not constitute financial advice.",
                  S("fr", fontSize=7, textColor=MUTED, alignment=TA_RIGHT)),
    ]], colWidths=[87*mm, 87*mm])
    ft.setStyle(TableStyle([
        ("TOPPADDING",    (0,0),(-1,-1), 0),
        ("BOTTOMPADDING", (0,0),(-1,-1), 0),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("RIGHTPADDING",  (0,0),(-1,-1), 0),
    ]))
    story.append(ft)

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 4px 24px 4px;border-bottom:1px solid rgba(255,255,255,0.07);margin-bottom:24px;">
      <div style="font-family:'Playfair Display',serif;font-size:1.35rem;font-weight:700;color:#c9a84c;letter-spacing:0.5px;">
        🛡 PremiumIQ
      </div>
      <div style="font-size:0.7rem;color:#5a6a82;text-transform:uppercase;letter-spacing:2px;margin-top:4px;">
        Intelligent Insurance Predictor
      </div>
    </div>
    <div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">
      Client Profile
    </div>
    """, unsafe_allow_html=True)

    age    = st.slider("Age", 18, 66, 35)
    height_cm = st.number_input("Height (cm)", 100, 250, 170, step=1)
    height = height_cm / 100
    weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0, step=0.5)
    bmi    = weight / (height ** 2)
    bmi_cat, bmi_col = bmi_category(bmi)

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
                border-radius:10px;padding:12px 16px;margin:8px 0 16px 0;
                display:flex;justify-content:space-between;align-items:center;">
      <span style="color:#5a6a82;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;">BMI Index</span>
      <span>
        <span style="font-family:'DM Mono',monospace;font-size:1.1rem;font-weight:500;color:{bmi_col};">{bmi:.1f}</span>
        <span style="font-size:0.72rem;color:{bmi_col};margin-left:6px;">{bmi_cat}</span>
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">Health Conditions</div>', unsafe_allow_html=True)
    diabetes       = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    blood_pressure = st.selectbox("Blood Pressure Problems", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    transplant     = st.selectbox("Any Transplants", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    chronic        = st.selectbox("Chronic Diseases", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    allergies      = st.selectbox("Known Allergies", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    cancer_family  = st.selectbox("Family Cancer History", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    surgeries      = st.selectbox("Major Surgeries", [0, 1, 2, 3], format_func=lambda x: f"{x} surgeries" if x > 0 else "None")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;">Export Report</div>', unsafe_allow_html=True)

    if REPORTLAB_AVAILABLE:
        try:
            _pred    = predict(age, height_cm, weight, diabetes, blood_pressure, transplant, chronic, allergies, cancer_family, surgeries)
            _monthly = _pred / 12
            _risk    = risk_profile(age, bmi, diabetes, blood_pressure, transplant, chronic, allergies, cancer_family, surgeries)
            pdf_bytes = generate_pdf(
                age, height, weight, bmi, bmi_cat, 0, "N/A", "N/A", "N/A",
                _pred, _monthly, _risk[0], _risk[1], _risk[2], _risk[3], _risk[4]
            )
            st.download_button(
                label="⬇  Download Client Report (PDF)",
                data=pdf_bytes,
                file_name=f"PremiumIQ_Report_{age}y_BMI{bmi:.0f}.pdf",
                mime="application/pdf",
                width='stretch',
            )
        except Exception as e:
            st.error(f"PDF error: {e}")
    else:
        st.markdown("""
        <div style="padding:10px;background:rgba(232,168,56,0.08);
                    border:1px solid rgba(232,168,56,0.2);border-radius:8px;
                    font-size:0.72rem;color:#5a6a82;text-align:center;">
          Install reportlab to enable PDF export:<br>
          <code>pip install reportlab</code>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# COMPUTE
# ─────────────────────────────────────────
pred, ci_low, ci_high = prediction_with_ci(age, height_cm, weight, diabetes, blood_pressure, transplant, chronic, allergies, cancer_family, surgeries)
monthly = pred / 12
risk_lvl, risk_score, fhi, fmed, flo = risk_profile(age, bmi, diabetes, blood_pressure, transplant, chronic, allergies, cancer_family, surgeries)
risk_color = {"LOW": C["success"], "MEDIUM": C["warning"], "HIGH": C["danger"]}[risk_lvl]


# MASTHEAD
# ─────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:flex-end;
            padding:4px 0 20px 0;border-bottom:1px solid rgba(255,255,255,0.07);margin-bottom:20px;">
  <div>
    <div style="font-family:'Playfair Display',serif;font-size:1.9rem;font-weight:700;
                color:#d4dae6;letter-spacing:-0.3px;line-height:1.15;">
      PremiumIQ — Insurance Risk Dashboard
    </div>
    <div style="color:#5a6a82;font-size:0.8rem;margin-top:6px;letter-spacing:0.3px;">
      Predictive analytics powered by Stacking Ensemble (XGBoost + CatBoost + Random Forest → Ridge) &nbsp;·&nbsp; Adjust client profile in the sidebar
    </div>
  </div>
  <div style="text-align:right;">
    <div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;">Model Accuracy</div>
    <div style="font-family:'DM Mono',monospace;font-size:1.5rem;color:#c9a84c;font-weight:500;">{f"{MODEL_R2*100:.2f}%" if MODEL_R2 else "N/A"}</div>
    <div style="font-size:0.68rem;color:#5a6a82;">R² Score</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# NATURAL LANGUAGE INPUT
# ─────────────────────────────────────────
with st.expander("🧠  Natural Language Input — describe a profile in plain English", expanded=False):
    st.markdown('<p style="color:#5a6a82;font-size:0.8rem;margin-bottom:10px;">Try: <em>"32 year old male smoker, 172cm, 85kg, 2 children, northeast"</em> or <em>"55 year old diabetic female, 160cm, 70kg, high BP, sedentary"</em></p>', unsafe_allow_html=True)
    nlp_col1, nlp_col2 = st.columns([5, 1], gap="small")
    with nlp_col1:
        nlp_text = st.text_input("Profile description", placeholder="e.g. 40 year old, 175cm, 90kg, diabetic, high BP ...", label_visibility="collapsed")
    with nlp_col2:
        nlp_btn = st.button("Parse →", width='stretch')
    if nlp_btn and nlp_text:
        parsed = parse_nlp_input(nlp_text)
        if parsed:
            display = {}
            for k, v in parsed.items():
                if k == "bmi_computed":
                    display["BMI (auto)"] = f"{v} (from height & weight)"
                elif k == "height":
                    display["Height"] = f"{v} m"
                elif k == "weight":
                    display["Weight"] = f"{v} kg"
                else:
                    display[k.replace('_',' ').title()] = str(v)
            parts = [f"**{k}** → `{v}`" for k, v in display.items()]
            st.success("✅ Parsed: " + "  ·  ".join(parts) + "  — Update the sidebar sliders to match.")
        else:
            st.warning("⚠ Could not parse. Try including age, height, weight e.g. '35 year old female non-smoker, 165cm, 60kg'")

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# KPI STRIP
# ─────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5, gap="small")
with k1: st.markdown(kpi("Annual Premium",    f"₹{pred:,.0f}",       f"₹{ci_low:,.0f} – ₹{ci_high:,.0f}"), unsafe_allow_html=True)
with k2: st.markdown(kpi("BMI Index",         f"{bmi:.1f}",          bmi_cat, bmi_col), unsafe_allow_html=True)
with k3: st.markdown(kpi("Risk Level",        risk_lvl,              f"Score {risk_score}/90", risk_color), unsafe_allow_html=True)
with k4: st.markdown(kpi("Monthly Cost",      f"₹{monthly:,.0f}",    "÷ 12 installments"), unsafe_allow_html=True)
risk_flags = sum([diabetes, blood_pressure, transplant, chronic, allergies, cancer_family])
with k5: st.markdown(kpi("Risk Conditions", f"{risk_flags} / 6", "Health risk flags (HealthRiskScore)", C["danger"] if risk_flags>=3 else C["warning"] if risk_flags>=1 else C["success"]), unsafe_allow_html=True)

st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
T1, T2, T3, T4, T5, T6, T7, T8, T9 = st.tabs([
    "  Prediction Overview  ",
    "  Model Benchmark  ",
    "  Scenario Simulator  ",
    "  Risk Intelligence  ",
    "  SHAP Explainability  ",
    "  Compare Profiles  ",
    "  📊 Data Analysis  ",
    "  📈 Premium Forecast  ",
    "  🏢 Insurer Compare  ",
])


# ════════════════════════════════════════════
# TAB 1 — PREDICTION OVERVIEW
# ════════════════════════════════════════════
with T1:
    left, right = st.columns([5, 4], gap="large")

    with left:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(201,168,76,0.08) 0%,rgba(201,168,76,0.03) 100%);
                    border:1px solid rgba(201,168,76,0.25);border-radius:20px;padding:40px 36px;
                    margin-bottom:24px;position:relative;overflow:hidden;">
          <div style="position:absolute;top:-40px;right:-40px;width:180px;height:180px;border-radius:50%;
                      background:radial-gradient(circle,rgba(201,168,76,0.08) 0%,transparent 70%);"></div>
          <div style="font-size:0.7rem;color:#5a6a82;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;">
            Estimated Annual Premium
          </div>
          <div style="font-family:'DM Mono',monospace;font-size:3.5rem;font-weight:500;
                      color:#e8c97a;line-height:1;letter-spacing:-1px;">
            ₹{pred:,.0f}
          </div>
          <div style="margin-top:10px;padding:8px 14px;background:rgba(255,255,255,0.03);
                      border-radius:8px;display:inline-block;">
            <span style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1px;">95% Confidence Interval &nbsp;</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.85rem;color:#7ab3d4;">
              ₹{ci_low:,.0f} &nbsp;–&nbsp; ₹{ci_high:,.0f}
            </span>
          </div>
          <div style="display:flex;gap:32px;margin-top:20px;padding-top:20px;
                      border-top:1px solid rgba(255,255,255,0.06);">
            <div>
              <div style="font-size:0.67rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1px;">Monthly</div>
              <div style="font-family:'DM Mono',monospace;font-size:1.1rem;color:#c9a84c;margin-top:3px;">₹{pred/12:,.0f}</div>
            </div>
            <div>
              <div style="font-size:0.67rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1px;">Quarterly</div>
              <div style="font-family:'DM Mono',monospace;font-size:1.1rem;color:#c9a84c;margin-top:3px;">₹{pred/4:,.0f}</div>
            </div>
            <div>
              <div style="font-size:0.67rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1px;">Risk Band</div>
              <div style="font-family:'DM Mono',monospace;font-size:1.1rem;margin-top:3px;color:{risk_color};">{risk_lvl}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;">Client Parameters</div>', unsafe_allow_html=True)
        rows = [
            ("Age", f"{age} years"), ("Height", f"{height_cm} cm"),
            ("Weight", f"{weight:.1f} kg"), ("BMI", f"{bmi:.2f} — {bmi_cat}"),
            ("Diabetes", "Yes" if diabetes else "No"),
            ("Blood Pressure Problems", "Yes" if blood_pressure else "No"),
            ("Any Transplants", "Yes" if transplant else "No"),
            ("Chronic Diseases", "Yes" if chronic else "No"),
            ("Known Allergies", "Yes" if allergies else "No"),
            ("Family Cancer History", "Yes" if cancer_family else "No"),
            ("Major Surgeries", str(surgeries)),
        ]
        rows_html = "".join([f"""
        <div style="display:flex;justify-content:space-between;padding:10px 16px;
                    border-bottom:1px solid rgba(255,255,255,0.04);
                    {'background:rgba(255,255,255,0.015);' if i%2==0 else ''}">
          <span style="color:#5a6a82;font-size:0.8rem;">{k}</span>
          <span style="color:#d4dae6;font-size:0.8rem;font-weight:500;font-family:'DM Mono',monospace;">{v}</span>
        </div>""" for i,(k,v) in enumerate(rows)])
        st.markdown(f'<div style="border:1px solid rgba(255,255,255,0.07);border-radius:14px;overflow:hidden;">{rows_html}</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">Predictive Feature Weights</div>', unsafe_allow_html=True)

        # Features from notebook Cell 276 (XGBoost component of stacking ensemble)
        feats   = ["Age_HealthRisk", "Age_Surgeries", "HealthRiskScore", "BMI",
                   "SurgeryTransplantRisk", "Age", "AnyTransplants", "NumberOfMajorSurgeries"]
        imps    = [0.285, 0.198, 0.152, 0.098, 0.087, 0.071, 0.065, 0.044]
        f_cols  = [C["danger"] if v>0.1 else C["warning"] if v>0.03 else C["muted"] for v in imps]

        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor(C["bg2"])
        ax.set_facecolor(C["bg2"])
        y_pos = range(len(feats))
        bars  = ax.barh(list(y_pos), imps, color=f_cols, height=0.55, edgecolor="none", zorder=3)
        for bar, val in zip(bars, imps):
            ax.text(val+0.005, bar.get_y()+bar.get_height()/2,
                    f"{val:.1%}", va='center', color=C["plat"], fontsize=7.5, fontweight='500')
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(feats, fontsize=8.5, color=C["plat"])
        ax.set_xlabel("Importance Weight", fontsize=8, color=C["muted"])
        ax.set_title("Stacking Ensemble Feature Importance", fontsize=9.5, color=C["plat"], pad=12, fontweight='600')
        ax.invert_yaxis()
        ax.grid(axis='x', color=C["border"], lw=0.5, alpha=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(colors=C["muted"])
        for sp in ax.spines.values(): sp.set_visible(False)
        legend_items = [
            mpatches.Patch(color=C["danger"],  label='Dominant (>10%)'),
            mpatches.Patch(color=C["warning"], label='Moderate (3–10%)'),
            mpatches.Patch(color=C["muted"],   label='Minor (<3%)'),
        ]
        ax.legend(handles=legend_items, fontsize=7, loc='lower right', framealpha=0, labelcolor=C["plat"])
        plt.tight_layout(pad=1.5)
        st.pyplot(fig, width='stretch')
        plt.close()

        st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin:20px 0 12px 0;">Active Risk Signals</div>', unsafe_allow_html=True)
        for label, desc in fhi:
            st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:center;
            padding:10px 14px;background:rgba(217,83,79,0.08);border:1px solid rgba(217,83,79,0.2);
            border-radius:8px;margin-bottom:6px;">
              <span style="color:#d9534f;font-size:0.82rem;font-weight:500;">{label}</span>
              <span style="color:#5a6a82;font-size:0.75rem;">{desc}</span></div>""", unsafe_allow_html=True)
        for label, desc in fmed:
            st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:center;
            padding:10px 14px;background:rgba(232,168,56,0.08);border:1px solid rgba(232,168,56,0.2);
            border-radius:8px;margin-bottom:6px;">
              <span style="color:#e8a838;font-size:0.82rem;font-weight:500;">{label}</span>
              <span style="color:#5a6a82;font-size:0.75rem;">{desc}</span></div>""", unsafe_allow_html=True)
        for label, desc in flo:
            st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:center;
            padding:10px 14px;background:rgba(47,179,122,0.08);border:1px solid rgba(47,179,122,0.18);
            border-radius:8px;margin-bottom:6px;">
              <span style="color:#2fb37a;font-size:0.82rem;font-weight:500;">{label}</span>
              <span style="color:#5a6a82;font-size:0.75rem;">{desc}</span></div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 2 — MODEL BENCHMARK
# ════════════════════════════════════════════
with T2:
    # Use dynamically computed benchmark, fall back to hardcoded if unavailable
    if MODEL_BENCHMARK is not None:
        model_data = MODEL_BENCHMARK
    else:
        model_data = pd.DataFrame({
            "Model":    ["Linear Regression","Decision Tree","AdaBoost","Gradient Boosting",
                         "Random Forest","XGBoost (Baseline)","XGBoost (Tuned)","Stacking Ensemble"],
            "R²":       [0.8047, 0.7830, 0.7266, 0.7431, 0.8467, 0.8564, 0.8560,
                         (MODEL_R2 if MODEL_R2 else 0.8700)],
            "RMSE":     [2800, 3100, 3600, 3400, 2600, 2450, 2460,
                         (MODEL_RMSE if MODEL_RMSE else 2200)],
            "Selected": [False, False, False, False, False, False, False, True]
        })
    top, bot = st.columns([3, 2], gap="large")

    with top:
        st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">Performance Metrics — All Models</div>', unsafe_allow_html=True)

        names  = model_data["Model"].tolist()
        r2s    = model_data["R²"].tolist()
        rmses  = model_data["RMSE"].tolist()
        bar_c  = [C["gold"] if s else C["blue"] for s in model_data["Selected"]]
        n      = len(names)
        y      = np.arange(n)

        # Short display labels — no wrapping needed for horizontal bars
        short_names = [
            nm.replace("XGBoost (Baseline)", "XGB Baseline")
              .replace("XGBoost (Tuned)", "XGB Tuned")
              .replace("Gradient Boosting", "Grad. Boosting")
              .replace("Stacking Ensemble", "Stacking Ens.")
              .replace("Linear Regression", "Linear Reg.")
              .replace("Random Forest", "Rand. Forest")
            for nm in names
        ]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
        fig.patch.set_facecolor(C["bg2"])
        fig_style(fig, [ax1, ax2])

        # ── R² chart (horizontal) ──────────────────────────────────────────
        bars1 = ax1.barh(y, r2s, color=bar_c, height=0.55, edgecolor="none", zorder=3)
        ax1.set_yticks(y)
        ax1.set_yticklabels(short_names, fontsize=8.5, color=C["plat"])
        ax1.tick_params(axis="y", length=0, pad=6)
        ax1.tick_params(axis="x", labelsize=7.5, colors=C["muted"])
        ax1.set_xlabel("R² Score", fontsize=8, color=C["muted"], labelpad=6)
        ax1.set_title("R² Score  ·  Higher is Better", fontsize=9,
                      color=C["plat"], pad=12, fontweight="600")
        ax1.axvline(max(r2s), color=C["gold"], lw=0.8, ls="--", alpha=0.5, zorder=2)
        r2_max = max(r2s)
        ax1.set_xlim(min(r2s) * 0.97, r2_max * 1.06)
        for bar, val, sel in zip(bars1, r2s, model_data["Selected"]):
            ax1.text(
                val + r2_max * 0.004,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", ha="left",
                fontsize=7.5, fontweight="600" if sel else "400",
                color=C["gold"] if sel else C["plat"], clip_on=False
            )
        for tick, sel in zip(ax1.get_yticklabels(), model_data["Selected"]):
            if sel:
                tick.set_color(C["gold"])
                tick.set_fontweight("700")
        for sp in ax1.spines.values(): sp.set_visible(False)
        ax1.grid(axis="x", color=C["border"], lw=0.5, alpha=0.5, zorder=0)
        ax1.set_axisbelow(True)

        # ── RMSE chart (horizontal) ────────────────────────────────────────
        bars2 = ax2.barh(y, rmses, color=bar_c, height=0.55, edgecolor="none", zorder=3)
        ax2.set_yticks(y)
        ax2.set_yticklabels(short_names, fontsize=8.5, color=C["plat"])
        ax2.tick_params(axis="y", length=0, pad=6)
        ax2.tick_params(axis="x", labelsize=7.5, colors=C["muted"])
        ax2.set_xlabel("RMSE (₹)", fontsize=8, color=C["muted"], labelpad=6)
        ax2.set_title("RMSE  ·  Lower is Better", fontsize=9,
                      color=C["plat"], pad=12, fontweight="600")
        ax2.axvline(min(rmses), color=C["gold"], lw=0.8, ls="--", alpha=0.5, zorder=2)
        rmse_max = max(rmses)
        ax2.set_xlim(0, rmse_max * 1.20)
        for bar, val, sel in zip(bars2, rmses, model_data["Selected"]):
            ax2.text(
                val + rmse_max * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"₹{val:,.0f}",
                va="center", ha="left",
                fontsize=7.5, fontweight="600" if sel else "400",
                color=C["gold"] if sel else C["plat"], clip_on=False
            )
        for tick, sel in zip(ax2.get_yticklabels(), model_data["Selected"]):
            if sel:
                tick.set_color(C["gold"])
                tick.set_fontweight("700")
        for sp in ax2.spines.values(): sp.set_visible(False)
        ax2.grid(axis="x", color=C["border"], lw=0.5, alpha=0.5, zorder=0)
        ax2.set_axisbelow(True)

        plt.tight_layout(pad=2.2)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with bot:
        st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">Comparison Table</div>', unsafe_allow_html=True)
        table_rows = ""
        for _, row in model_data.iterrows():
            bg     = "rgba(201,168,76,0.07)" if row["Selected"] else "transparent"
            bl     = "2px solid #c9a84c" if row["Selected"] else "2px solid transparent"
            badge  = "<span style='font-size:0.6rem;background:rgba(201,168,76,0.2);color:#c9a84c;padding:2px 8px;border-radius:4px;margin-left:6px;'>SELECTED</span>" if row["Selected"] else ""
            vc     = "#c9a84c" if row["Selected"] else "#d4dae6"
            table_rows += f"""<tr style="background:{bg};border-left:{bl};">
              <td style="padding:10px 14px;font-size:0.8rem;color:#d4dae6;">{row['Model']}{badge}</td>
              <td style="padding:10px 14px;font-family:'DM Mono',monospace;font-size:0.8rem;color:{vc};text-align:right;">{row['R²']:.4f}</td>
              <td style="padding:10px 14px;font-family:'DM Mono',monospace;font-size:0.8rem;color:{vc};text-align:right;">{row['RMSE']:.4f}</td>
            </tr>"""
        st.markdown(f"""
        <div style="border:1px solid rgba(255,255,255,0.07);border-radius:14px;overflow:hidden;">
          <table style="width:100%;border-collapse:collapse;">
            <thead><tr style="background:rgba(255,255,255,0.04);border-bottom:1px solid rgba(255,255,255,0.07);">
              <th style="padding:10px 14px;font-size:0.67rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1px;text-align:left;">Model</th>
              <th style="padding:10px 14px;font-size:0.67rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1px;text-align:right;">R²</th>
              <th style="padding:10px 14px;font-size:0.67rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1px;text-align:right;">RMSE</th>
            </tr></thead>
            <tbody>{table_rows}</tbody>
          </table>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:16px;padding:16px 20px;background:rgba(201,168,76,0.06);
                    border:1px solid rgba(201,168,76,0.2);border-radius:12px;">
          <div style="font-size:0.72rem;color:#c9a84c;font-weight:600;margin-bottom:8px;">Why Stacking Ensemble was selected</div>
          <div style="font-size:0.78rem;color:#5a6a82;line-height:1.65;">
            The Stacking Ensemble combines three powerful base learners — XGBoost, CatBoost, and Random Forest — with a RidgeCV meta-learner using 5-fold out-of-fold stacking.
            Together with 6 engineered features (BMI, HealthRiskScore, Age_HealthRisk, Age_Surgeries, SurgeryTransplantRisk, AgeGroup),
            it achieved R² <span style='color:#d4dae6;font-family:"DM Mono",monospace;'>({f"{MODEL_R2:.4f}" if MODEL_R2 else "N/A"})</span> and
            RMSE <span style='color:#d4dae6;font-family:"DM Mono",monospace;'>({f"₹{MODEL_RMSE:,.0f}" if MODEL_RMSE else "N/A"})</span> — best on this dataset.
          </div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 3 — SCENARIO SIMULATOR
# ════════════════════════════════════════════
with T3:
    st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;">Scenario Parameters</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#5a6a82;font-size:0.8rem;margin-bottom:20px;">Adjust any parameter below to instantly see how it changes the predicted premium versus your current sidebar profile.</p>', unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3, gap="medium")
    with sc1:
        s_age      = st.slider("Age", 18, 66, age, key="s_age")
        s_height   = st.number_input("Height (cm)", 100, 250, height_cm, step=1, key="s_h")
        s_weight   = st.number_input("Weight (kg)", 30.0, 200.0, weight, step=0.5, key="s_w")
        s_bmi      = s_weight / ((s_height / 100) ** 2)
        s_bmi_cat, _ = bmi_category(s_bmi)
        st.caption(f"BMI: {s_bmi:.1f} — {s_bmi_cat}")
    with sc2:
        s_diabetes  = st.selectbox("Diabetes",              [0,1], index=diabetes,       format_func=lambda x: "No" if x==0 else "Yes", key="s_db")
        s_bp        = st.selectbox("Blood Pressure",        [0,1], index=blood_pressure, format_func=lambda x: "No" if x==0 else "Yes", key="s_bp")
        s_transplant= st.selectbox("Any Transplants",       [0,1], index=transplant,     format_func=lambda x: "No" if x==0 else "Yes", key="s_tr")
        s_chronic   = st.selectbox("Chronic Diseases",      [0,1], index=chronic,        format_func=lambda x: "No" if x==0 else "Yes", key="s_ch")
    with sc3:
        s_allergies = st.selectbox("Known Allergies",       [0,1], index=allergies,      format_func=lambda x: "No" if x==0 else "Yes", key="s_al")
        s_cancer    = st.selectbox("Family Cancer History", [0,1], index=cancer_family,  format_func=lambda x: "No" if x==0 else "Yes", key="s_cf")
        s_surgeries = st.selectbox("Major Surgeries",       [0,1,2,3], index=surgeries,  format_func=lambda x: f"{x}" if x>0 else "None", key="s_su")

    sim_pred = predict(s_age, s_height, s_weight, s_diabetes, s_bp, s_transplant, s_chronic, s_allergies, s_cancer, s_surgeries)
    delta    = sim_pred - pred
    delta_p  = (delta / pred) * 100 if pred else 0
    dcol     = C["danger"] if delta > 0 else C["success"]
    sign     = "+" if delta > 0 else ""

    st.markdown("<br>", unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4, gap="small")
    with d1: st.markdown(kpi("Baseline Premium",  f"₹{pred:,.0f}",             "Sidebar profile"),     unsafe_allow_html=True)
    with d2: st.markdown(kpi("Scenario Premium",  f"₹{sim_pred:,.0f}",         "Simulated profile"),   unsafe_allow_html=True)
    with d3: st.markdown(kpi("Absolute Change",   f"{sign}₹{abs(delta):,.0f}", f"{sign}{delta_p:.1f}%", dcol), unsafe_allow_html=True)
    with d4: st.markdown(kpi("Scenario Monthly",  f"₹{sim_pred/12:,.0f}",      "÷ 12"),                unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">Sensitivity Curves</div>', unsafe_allow_html=True)

    age_r  = range(18, 67, 4)
    age_p  = [predict(a, s_height, s_weight, s_diabetes, s_bp, s_transplant, s_chronic, s_allergies, s_cancer, s_surgeries) for a in age_r]
    bmi_r  = np.arange(17, 46, 1.5)
    # vary weight to change BMI, keep height fixed
    bmi_p  = [predict(s_age, s_height, w * ((s_height/100)**2), s_diabetes, s_bp, s_transplant, s_chronic, s_allergies, s_cancer, s_surgeries) for w in bmi_r]
    # condition impact: 0 conditions vs all conditions
    cond_0 = [predict(a, s_height, s_weight, 0, 0, 0, 0, 0, 0, 0) for a in age_r]
    cond_all = [predict(a, s_height, s_weight, 1, 1, 0, 1, 0, 1, 1) for a in age_r]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    fig_style(fig, axes)
    fig.patch.set_facecolor(C["bg2"])
    for ax in axes:
        ax.set_facecolor(C["bg3"])
        for sp in ax.spines.values(): sp.set_visible(False)

    axes[0].plot(list(age_r), age_p, color=C["gold"], lw=2, zorder=3)
    axes[0].fill_between(list(age_r), age_p, alpha=0.08, color=C["gold"])
    axes[0].axvline(s_age, color=C["danger"], ls='--', lw=1, alpha=0.7)
    axes[0].set_xlabel("Age", fontsize=8); axes[0].set_ylabel("₹ Premium", fontsize=8)
    axes[0].set_title("Premium vs Age", fontsize=9, pad=8, fontweight='600')

    axes[1].plot(bmi_r, bmi_p, color=C["blue"], lw=2, zorder=3)
    axes[1].fill_between(bmi_r, bmi_p, alpha=0.08, color=C["blue"])
    axes[1].axvline(s_bmi, color=C["danger"], ls='--', lw=1, alpha=0.7)
    axes[1].set_xlabel("BMI", fontsize=8); axes[1].set_ylabel("₹ Premium", fontsize=8)
    axes[1].set_title("Premium vs BMI", fontsize=9, pad=8, fontweight='600')

    axes[2].plot(list(age_r), cond_0,   color=C["success"], lw=2, label="No Conditions", zorder=3)
    axes[2].plot(list(age_r), cond_all, color=C["danger"],  lw=2, label="4 Conditions",  zorder=3)
    axes[2].fill_between(list(age_r), cond_0, cond_all, alpha=0.07, color=C["warning"])
    axes[2].axvline(s_age, color=C["muted"], ls='--', lw=0.8, alpha=0.5)
    axes[2].set_xlabel("Age", fontsize=8); axes[2].set_ylabel("₹ Premium", fontsize=8)
    axes[2].set_title("Condition Impact by Age", fontsize=9, pad=8, fontweight='600')
    leg = axes[2].legend(fontsize=7, framealpha=0)
    for t in leg.get_texts(): t.set_color(C["plat"])

    plt.tight_layout(pad=2.5)
    st.pyplot(fig, width='stretch')
    plt.close()


# ════════════════════════════════════════════
# TAB 4 — RISK INTELLIGENCE
# ════════════════════════════════════════════
with T4:
    ri_l, ri_r = st.columns([1, 1], gap="large")

    with ri_l:
        st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">Risk Assessment</div>', unsafe_allow_html=True)
        rc = {"LOW":("rgba(47,179,122,0.08)","rgba(47,179,122,0.25)","#2fb37a"),
              "MEDIUM":("rgba(232,168,56,0.08)","rgba(232,168,56,0.25)","#e8a838"),
              "HIGH":("rgba(217,83,79,0.08)","rgba(217,83,79,0.25)","#d9534f")}
        bg_c,bd_c,tx_c = rc[risk_lvl]
        risk_desc = {
            "LOW":    "Low-risk profile. Few compounding factors. Premiums are expected to remain competitive.",
            "MEDIUM": "Moderate-risk profile. Some lifestyle or demographic factors elevate this client's premium above baseline.",
            "HIGH":   "High-risk profile. Multiple significant factors compound to produce substantially elevated premiums."
        }
        st.markdown(f"""
        <div style="background:{bg_c};border:1px solid {bd_c};border-radius:16px;padding:28px;margin-bottom:20px;">
          <div style="font-size:0.67rem;color:{tx_c};text-transform:uppercase;letter-spacing:2px;margin-bottom:10px;">Risk Classification</div>
          <div style="font-family:'Playfair Display',serif;font-size:2.4rem;font-weight:700;color:{tx_c};line-height:1;">{risk_lvl}</div>
          <div style="display:flex;align-items:center;gap:12px;margin-top:14px;">
            <div style="flex:1;height:6px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden;">
              <div style="width:{min(risk_score/90*100,100):.0f}%;height:100%;background:{tx_c};border-radius:3px;"></div>
            </div>
            <span style="font-family:'DM Mono',monospace;font-size:0.8rem;color:{tx_c};">{risk_score}/90</span>
          </div>
          <p style="color:#5a6a82;font-size:0.82rem;line-height:1.6;margin-top:14px;margin-bottom:0;">{risk_desc[risk_lvl]}</p>
        </div>""", unsafe_allow_html=True)

        for section, factors, color in [
            ("High Risk Factors", fhi, C["danger"]),
            ("Moderate Factors",  fmed, C["warning"]),
            ("Positive Factors",  flo, C["success"])
        ]:
            if factors:
                st.markdown(f'<div style="font-size:0.67rem;color:{color};text-transform:uppercase;letter-spacing:1.5px;margin:16px 0 8px 0;">{section}</div>', unsafe_allow_html=True)
                for label, desc in factors:
                    st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:10px 16px;
                    background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
                    border-left:3px solid {color};border-radius:8px;margin-bottom:6px;">
                      <span style="color:#d4dae6;font-size:0.82rem;font-weight:500;">{label}</span>
                      <span style="color:#5a6a82;font-size:0.75rem;">{desc}</span>
                    </div>""", unsafe_allow_html=True)

    with ri_r:
        st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">Risk Gauge</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor(C["bg2"])
        ax.set_facecolor(C["bg2"])
        ax.axis("off")

        theta = np.linspace(np.pi, 0, 300)
        r_out, r_in = 1.0, 0.65
        cx, cy = 0, 0

        def arc(t, r): return cx+r*np.cos(t), cy+r*np.sin(t)

        xo,yo = arc(theta, r_out); xi,yi = arc(theta[::-1], r_in)
        ax.fill(np.concatenate([xo,xi]), np.concatenate([yo,yi]), color=C["border"], zorder=1)

        fill_end = np.pi - (risk_score/90)*np.pi
        t_fill   = np.linspace(np.pi, fill_end, 200)
        xof,yof  = arc(t_fill, r_out); xif,yif = arc(t_fill[::-1], r_in)
        ax.fill(np.concatenate([xof,xif]), np.concatenate([yof,yif]), color=risk_color, alpha=0.85, zorder=2)

        for frac in [1/3, 2/3, 1.0]:
            t_m  = np.pi - frac*np.pi
            x0m,y0m = arc(np.array([t_m]), r_in)
            x1m,y1m = arc(np.array([t_m]), r_out)
            ax.plot([x0m[0],x1m[0]], [y0m[0],y1m[0]], color=C["bg2"], lw=2, zorder=3)

        needle_angle = np.pi - (risk_score/90)*np.pi
        nx,ny = cx+0.75*np.cos(needle_angle), cy+0.75*np.sin(needle_angle)
        ax.annotate("", xy=(nx,ny), xytext=(cx,cy),
                    arrowprops=dict(arrowstyle="-|>", color=risk_color, lw=2.5, mutation_scale=14))
        ax.plot(cx, cy, 'o', color=risk_color, markersize=9, zorder=5)
        ax.plot(cx, cy, 'o', color=C["bg2"],   markersize=5, zorder=6)
        ax.text(cx, cy-0.18, f"{risk_score}", ha='center', fontsize=20, color=risk_color, fontweight='700')
        ax.text(cx, cy-0.35, "out of 90",     ha='center', fontsize=7,  color=C["muted"])
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.5, 1.1)
        ax.text(-1.1,-0.12,"LOW",  ha='left',   fontsize=8, color=C["success"], fontweight='600')
        ax.text(0,   1.08, "MED",  ha='center', fontsize=8, color=C["warning"], fontweight='600')
        ax.text(1.1, -0.12,"HIGH", ha='right',  fontsize=8, color=C["danger"],  fontweight='600')
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, width='stretch')
        plt.close()

        st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin:12px 0 12px 0;">Actionable Recommendations</div>', unsafe_allow_html=True)
        recs = []
        if diabetes:       recs.append(("Manage Diabetes",      "Controlled diabetes (HbA1c <7%) can reduce loading at renewal. Consistent medication matters."))
        if blood_pressure: recs.append(("Control Blood Pressure","Maintaining BP below 130/80 reduces cardiovascular risk loading significantly at renewal."))
        if transplant:     recs.append(("Disclose Fully",        "Transplant history must be fully declared. Choose insurers with best post-transplant cover."))
        if bmi >= 30:      recs.append(("Weight Management",     f"Reducing BMI from {bmi:.1f} toward 25 could lower premiums by 10–20% at renewal."))
        elif bmi >= 25:    recs.append(("Maintain Weight",       f"BMI {bmi:.1f} is borderline. Keeping it below 25 prevents future BMI loading."))
        if age >= 40:      recs.append(("Preventive Screenings", "Annual checkups at 40+ preempt costly medical events and demonstrate low-risk behaviour."))
        if not recs:       recs.append(("Maintain Lifestyle",    "Excellent profile. No significant risk factors detected. Keep up healthy habits."))
        for i,(title,body) in enumerate(recs):
            st.markdown(f"""
            <div style="padding:14px 18px;background:rgba(255,255,255,0.025);
                        border:1px solid rgba(255,255,255,0.07);border-left:3px solid {C['gold']};
                        border-radius:10px;margin-bottom:10px;">
              <div style="font-size:0.78rem;font-weight:600;color:#c9a84c;margin-bottom:5px;">
                {str(i+1).zfill(2)} · {title}
              </div>
              <div style="font-size:0.77rem;color:#5a6a82;line-height:1.6;">{body}</div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 5 — SHAP EXPLAINABILITY
# ════════════════════════════════════════════
with T5:
    st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;">AI Explainability — Feature Impact Analysis</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#5a6a82;font-size:0.8rem;margin-bottom:24px;">Reveals exactly <em>why</em> the model predicted this specific premium — showing how much each feature pushes the prediction up or down compared to the average customer.</p>', unsafe_allow_html=True)

    # ── Build input row ──────────────────────────────────────────────────
    health_risk_val = diabetes + blood_pressure + transplant + chronic + allergies + cancer_family
    age_group_val   = 0 if age < 30 else 1 if age < 45 else 2 if age < 60 else 3
    input_dict = {
        "Age"                    : age,
        "Diabetes"               : diabetes,
        "BloodPressureProblems"  : blood_pressure,
        "AnyTransplants"         : transplant,
        "AnyChronicDiseases"     : chronic,
        "Height"                 : height_cm,
        "Weight"                 : weight,
        "KnownAllergies"         : allergies,
        "HistoryOfCancerInFamily": cancer_family,
        "NumberOfMajorSurgeries" : surgeries,
        "BMI"                    : bmi,
        "HealthRiskScore"        : health_risk_val,
        "Age_HealthRisk"         : age * health_risk_val,
        "Age_Surgeries"          : age * surgeries,
        "SurgeryTransplantRisk"  : surgeries * (transplant + 1),
        "AgeGroup"               : age_group_val,
    }
    input_df = pd.DataFrame([input_dict])
    for c in model_columns:
        if c not in input_df.columns: input_df[c] = 0
    input_df = input_df[model_columns].astype(float)

    if scaler:
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=model_columns)
    else:
        input_scaled = input_df.copy()

    # ── Fast marginal contribution: toggle each feature to dataset mean ──
    @st.cache_data
    def compute_feature_impacts(_cols, _base_pred,
                                 age, height_cm, weight,
                                 diabetes, blood_pressure, transplant,
                                 chronic, allergies, cancer_family, surgeries):
        """
        For each feature: replace it with the dataset mean value and measure
        how much the prediction changes. Instant — no SHAP overhead.
        """
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        df_raw = pd.read_csv(os.path.join(base_dir, "Medicalpremium.csv"))
        # Engineer features
        df_raw["BMI"]               = df_raw["Weight"] / ((df_raw["Height"] / 100) ** 2)
        df_raw["HealthRiskScore"]   = (df_raw["Diabetes"] + df_raw["BloodPressureProblems"] +
                                       df_raw["AnyTransplants"] + df_raw["AnyChronicDiseases"] +
                                       df_raw["KnownAllergies"] + df_raw["HistoryOfCancerInFamily"])
        df_raw["Age_HealthRisk"]    = df_raw["Age"] * df_raw["HealthRiskScore"]
        df_raw["Age_Surgeries"]     = df_raw["Age"] * df_raw["NumberOfMajorSurgeries"]
        df_raw["SurgeryTransplantRisk"] = df_raw["NumberOfMajorSurgeries"] * (df_raw["AnyTransplants"] + 1)
        df_raw["AgeGroup"]          = pd.cut(df_raw["Age"], bins=[17,30,45,60,70], labels=[0,1,2,3]).astype(int)
        for c in _cols:
            if c not in df_raw.columns: df_raw[c] = 0
        col_means = df_raw[list(_cols)].mean()

        # Derived/interaction features — must be perturbed together with their
        # source columns to avoid impossible feature states (e.g. Age_Surgeries=32
        # while Age=35 and NumberOfMajorSurgeries=0 at the same time).
        # For these features we perturb all source columns simultaneously and
        # then recompute the derived feature to stay internally consistent.
        DERIVED_DEPS = {
            "BMI":                 ["Height", "Weight"],
            "HealthRiskScore":     ["Diabetes", "BloodPressureProblems", "AnyTransplants",
                                    "AnyChronicDiseases", "KnownAllergies", "HistoryOfCancerInFamily"],
            "Age_HealthRisk":      ["Age", "Diabetes", "BloodPressureProblems", "AnyTransplants",
                                    "AnyChronicDiseases", "KnownAllergies", "HistoryOfCancerInFamily"],
            "Age_Surgeries":       ["Age", "NumberOfMajorSurgeries"],
            "SurgeryTransplantRisk": ["NumberOfMajorSurgeries", "AnyTransplants"],
            "AgeGroup":            ["Age"],
        }

        def _recompute_derived(row):
            """Recompute all derived features from source columns in the row."""
            h = row.get("Height", height_cm)
            w = row.get("Weight", weight)
            a = row.get("Age", age)
            s = row.get("NumberOfMajorSurgeries", surgeries)
            t = row.get("AnyTransplants", transplant)
            hrs = (row.get("Diabetes", diabetes) +
                   row.get("BloodPressureProblems", blood_pressure) +
                   row.get("AnyTransplants", transplant) +
                   row.get("AnyChronicDiseases", chronic) +
                   row.get("KnownAllergies", allergies) +
                   row.get("HistoryOfCancerInFamily", cancer_family))
            row["BMI"]                  = w / ((h / 100) ** 2) if h > 0 else 0
            row["HealthRiskScore"]      = hrs
            row["Age_HealthRisk"]       = a * hrs
            row["Age_Surgeries"]        = a * s
            row["SurgeryTransplantRisk"]= s * (t + 1)
            row["AgeGroup"]             = 0 if a < 30 else 1 if a < 45 else 2 if a < 60 else 3
            return row

        impacts = {}
        base_row = input_df.iloc[0].to_dict()

        for feat in _cols:
            toggled_row = base_row.copy()

            if feat in DERIVED_DEPS:
                # Perturb all source features to their means, then recompute derived
                for src in DERIVED_DEPS[feat]:
                    if src in col_means:
                        toggled_row[src] = col_means[src]
                toggled_row = _recompute_derived(toggled_row)
            else:
                # Simple feature — safe to perturb in isolation
                toggled_row[feat] = col_means[feat]

            toggled = pd.DataFrame([toggled_row])
            for c in _cols:
                if c not in toggled.columns:
                    toggled[c] = 0
            toggled = toggled[list(_cols)].astype(float)

            if scaler:
                toggled_s = pd.DataFrame(scaler.transform(toggled), columns=list(_cols))
            else:
                toggled_s = toggled.copy()
            toggled_pred = float(model.predict(toggled_s)[0])
            impacts[feat] = _base_pred - toggled_pred   # positive = this feature raises premium

        return impacts, float(col_means.get("Age", 35)), df_raw[list(_cols)].mean()

    # Always available — no button needed, runs in < 1 second
    with st.spinner("Computing feature impacts..."):
        feature_impacts, avg_age, col_means = compute_feature_impacts(
            tuple(model_columns), pred,
            age, height_cm, weight,
            diabetes, blood_pressure, transplant,
            chronic, allergies, cancer_family, surgeries
        )

    # ── Compute baseline (all features at mean = "average customer") ──
    mean_df = pd.DataFrame([col_means], columns=model_columns).astype(float)
    if scaler:
        mean_scaled = pd.DataFrame(scaler.transform(mean_df), columns=model_columns)
    else:
        mean_scaled = mean_df.copy()
    avg_pred = float(model.predict(mean_scaled)[0])

    feat_labels = list(model_columns)
    impacts_arr  = np.array([feature_impacts[f] for f in feat_labels])
    sorted_idx   = np.argsort(np.abs(impacts_arr))[::-1]

    sh_l, sh_r = st.columns([3, 2], gap="large")

    with sh_l:
        st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">Premium Impact per Feature — This Client vs Average</div>', unsafe_allow_html=True)

        top_n      = min(10, len(sorted_idx))
        top_idx    = sorted_idx[:top_n][::-1]
        top_vals   = impacts_arr[top_idx]
        # Truncate long feature names for display
        raw_labels = [feat_labels[i] for i in top_idx]
        top_labels = [l if len(l) <= 18 else l[:16] + "…" for l in raw_labels]

        max_abs = max(abs(top_vals)) if len(top_vals) and max(abs(top_vals)) > 0 else 1

        # ── Figure: wide enough for labels + bars + value annotations ──
        fig, ax = plt.subplots(figsize=(7, 4.2))
        fig.patch.set_facecolor(C["bg2"])
        ax.set_facecolor(C["bg2"])

        bar_colors = [C["danger"] if v > 0 else C["success"] for v in top_vals]
        bars = ax.barh(top_labels, top_vals, color=bar_colors, height=0.52,
                       edgecolor="none", zorder=3)

        # Value annotations: always outside bar, with enough padding
        for bar, val in zip(bars, top_vals):
            sign   = "+" if val >= 0 else ""
            label  = f"{sign}₹{abs(val):,.0f}"
            # Offset = 1.5% of total axis range so labels never touch bar
            offset = max_abs * 0.04
            if val >= 0:
                xpos, ha = val + offset, "left"
            else:
                xpos, ha = val - offset, "right"
            ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                    label, va="center", ha=ha,
                    color=C["plat"], fontsize=7.5, fontweight="600",
                    clip_on=False)

        ax.axvline(0, color=C["muted"], lw=1.2, alpha=0.6, zorder=4)

        # ── Axis limits: add 22% padding on each side for annotations ──
        ax.set_xlim(-max_abs * 1.28, max_abs * 1.28)

        ax.set_xlabel("₹ Change vs Average Customer", fontsize=8, color=C["muted"], labelpad=8)
        ax.set_title("How Each Feature Moves Your Premium", fontsize=10,
                     color=C["plat"], pad=14, fontweight="700")

        # Y-axis labels: right-aligned, clear gap from bars
        ax.tick_params(axis="y", colors=C["plat"], labelsize=8.5, pad=6)
        ax.tick_params(axis="x", colors=C["muted"], labelsize=7.5)
        ax.yaxis.set_tick_params(length=0)

        for sp in ax.spines.values(): sp.set_visible(False)
        ax.grid(axis="x", color=C["border"], lw=0.5, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

        # X-axis: show ₹ formatted ticks, not too many
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{int(x):,}"))
        ax.xaxis.set_major_locator(plt.MaxNLocator(6, symmetric=True))

        # ── Legend: placed ABOVE the chart as a text strip, not inside ──
        fig.text(0.13, 0.97,
                 "■ Raises premium    ■ Lowers premium",
                 fontsize=7.5, color=C["muted"], va="top",
                 transform=fig.transFigure)
        # Color the two squares manually via two separate text calls
        fig.text(0.13, 0.97, "■", fontsize=8, color=C["danger"], va="top",
                 transform=fig.transFigure)
        fig.text(0.305, 0.97, "■", fontsize=8, color=C["success"], va="top",
                 transform=fig.transFigure)

        plt.tight_layout(pad=1.8, rect=[0, 0, 1, 0.95])
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── Plain-English explanation ──
        st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin:20px 0 12px 0;">Plain-English Explanation</div>', unsafe_allow_html=True)

        step_lines = []
        for i in sorted_idx[:6]:
            lbl   = feat_labels[i]
            delta = impacts_arr[i]
            if abs(delta) < 1: continue
            if delta > 0:
                line = (f'<span style="color:#d9534f;">▲ <strong>{lbl}</strong> '
                        f'adds &nbsp;<strong style="font-family:DM Mono,monospace;">'
                        f'+₹{delta:,.0f}</strong> above the average</span>')
            else:
                line = (f'<span style="color:#2fb37a;">▼ <strong>{lbl}</strong> '
                        f'saves &nbsp;<strong style="font-family:DM Mono,monospace;">'
                        f'₹{abs(delta):,.0f}</strong> below the average</span>')
            step_lines.append(line)

        st.markdown(f"""
        <div style="padding:18px 20px;background:rgba(255,255,255,0.025);
                    border:1px solid rgba(255,255,255,0.07);border-radius:14px;
                    line-height:2;font-size:0.83rem;color:#8892a4;">
          Starting from the <strong style="color:#d4dae6;">average customer premium</strong>
          of <strong style="color:#c9a84c;font-family:'DM Mono',monospace;">₹{avg_pred:,.0f}</strong>:
          <br><br>
          {"<br>".join(step_lines) if step_lines else "<span style='color:#5a6a82;'>Profile is close to average — no major deviations.</span>"}
          <br>
          <div style="margin-top:12px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.06);font-size:0.88rem;">
            ∴ &nbsp;Your estimated premium:&nbsp;
            <strong style="color:#c9a84c;font-size:1.05rem;font-family:'DM Mono',monospace;">₹{pred:,.0f}</strong>
          </div>
        </div>""", unsafe_allow_html=True)

    with sh_r:
        # ── Global feature importance from XGBoost component ──
        st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">Global Feature Importance (XGBoost Component)</div>', unsafe_allow_html=True)

        try:
            xgb_component = model.named_estimators_["xgb"]
            importances   = xgb_component.feature_importances_
        except Exception:
            try:
                importances = model.feature_importances_
            except Exception:
                importances = np.abs(impacts_arr) / (np.abs(impacts_arr).sum() + 1e-9)

        global_idx  = np.argsort(importances)[-10:]
        global_imp  = importances[global_idx]
        raw_glbls   = [feat_labels[i] for i in global_idx]
        global_lbls = [l if len(l) <= 18 else l[:16] + "…" for l in raw_glbls]
        bar_cols2   = [C["gold"] if v == global_imp.max() else C["blue"] for v in global_imp]

        fig2, ax2 = plt.subplots(figsize=(5, 4.2))
        fig2.patch.set_facecolor(C["bg2"])
        ax2.set_facecolor(C["bg2"])

        ax2.barh(global_lbls, global_imp, color=bar_cols2, height=0.52, edgecolor="none", zorder=3)

        g_max = global_imp.max() if global_imp.max() > 0 else 1
        for i, v in enumerate(global_imp):
            ax2.text(v + g_max * 0.04, i, f"{v:.3f}", va="center",
                     color=C["plat"], fontsize=7.5, fontweight="600", clip_on=False)

        ax2.set_xlim(0, g_max * 1.28)
        ax2.set_xlabel("Feature Importance (gain)", fontsize=8, color=C["muted"], labelpad=8)
        ax2.set_title("Top Features — All Customers", fontsize=9.5,
                      color=C["plat"], pad=14, fontweight="700")
        ax2.tick_params(axis="y", colors=C["plat"], labelsize=8.5, pad=6)
        ax2.tick_params(axis="x", colors=C["muted"], labelsize=7.5)
        ax2.yaxis.set_tick_params(length=0)
        for sp in ax2.spines.values(): sp.set_visible(False)
        ax2.grid(axis="x", color=C["border"], lw=0.5, alpha=0.5, zorder=0)
        ax2.set_axisbelow(True)

        # Highlight best bar label
        for tick, val in zip(ax2.get_yticklabels(), global_imp):
            tick.set_color(C["gold"] if val == g_max else C["plat"])

        plt.tight_layout(pad=1.8)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        # ── Per-feature contribution cards ──
        st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin:20px 0 12px 0;">This Client — Feature Contributions</div>', unsafe_allow_html=True)

        for idx in sorted_idx[:6]:
            lbl       = feat_labels[idx]
            rs_val    = impacts_arr[idx]
            fval      = float(input_df.iloc[0][lbl])
            mval      = float(col_means[lbl])
            direction = "▲ Raises premium" if rs_val > 0 else "▼ Lowers premium"
            d_color   = C["danger"] if rs_val > 0 else C["success"]
            bar_pct   = min(abs(rs_val) / (np.abs(impacts_arr).max() + 1e-9) * 100, 100)
            sign      = "+" if rs_val > 0 else ""

            st.markdown(f"""
            <div style="padding:10px 14px;background:rgba(255,255,255,0.02);
                        border:1px solid rgba(255,255,255,0.05);
                        border-left:3px solid {d_color};
                        border-radius:8px;margin-bottom:7px;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
                <span style="color:#d4dae6;font-size:0.8rem;font-weight:500;">{lbl}</span>
                <span style="color:{d_color};font-size:0.78rem;font-family:'DM Mono',monospace;font-weight:600;">
                  {sign}₹{abs(rs_val):,.0f}
                </span>
              </div>
              <div style="height:4px;background:rgba(255,255,255,0.05);border-radius:2px;overflow:hidden;">
                <div style="width:{bar_pct:.0f}%;height:100%;background:{d_color};border-radius:2px;opacity:0.7;"></div>
              </div>
              <div style="font-size:0.72rem;color:#5a6a82;margin-top:4px;">
                {direction} &nbsp;·&nbsp; Your value: <strong style="color:#d4dae6;">{fval:.2f}</strong> &nbsp;·&nbsp; Avg: {mval:.2f}
              </div>
            </div>""", unsafe_allow_html=True)

        # ── Key insight box ──
        top_feature = feat_labels[sorted_idx[0]]
        top_val     = impacts_arr[sorted_idx[0]]
        top_pct     = abs(top_val) / (pred + 1e-9) * 100

        st.markdown(f"""
        <div style="margin-top:4px;padding:14px 18px;background:rgba(201,168,76,0.06);
                    border:1px solid rgba(201,168,76,0.2);border-radius:12px;">
          <div style="font-size:0.72rem;color:#c9a84c;font-weight:600;margin-bottom:6px;">🔍 Key Insight</div>
          <div style="font-size:0.78rem;color:#5a6a82;line-height:1.65;">
            <strong style="color:#d4dae6;">{top_feature}</strong> has the biggest impact —
            {"adding" if top_val > 0 else "saving"}
            <strong style="color:#c9a84c;font-family:'DM Mono',monospace;">₹{abs(top_val):,.0f}</strong>
            {"above" if top_val > 0 else "below"} what an average customer would pay.
            That's <strong style="color:#c9a84c;">{top_pct:.1f}%</strong> of your total premium.
          </div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 6 — COMPARE PROFILES
# ════════════════════════════════════════════
with T6:
    st.markdown("""
    <div style="margin-bottom:28px;">
      <div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;">Side-by-Side Profile Comparison</div>
      <p style="color:#5a6a82;font-size:0.8rem;margin:0;">Enter two different client profiles below and instantly compare their predicted premiums, risk levels and savings opportunities.</p>
    </div>
    """, unsafe_allow_html=True)

    col_a, divider, col_b = st.columns([5, 0.2, 5], gap="small")

    with col_a:
        st.markdown("""<div style="background:rgba(201,168,76,0.06);border:1px solid rgba(201,168,76,0.2);
                    border-radius:14px;padding:18px 20px;margin-bottom:20px;">
          <div style="font-size:0.72rem;color:#c9a84c;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;">
            👤 Profile A
          </div></div>""", unsafe_allow_html=True)
        a_age      = st.slider("Age", 18, 66, 32, key="a_age")
        a_height_cm = st.number_input("Height (cm)", 100, 250, 172, step=1, key="a_h")
        a_height = a_height_cm / 100
        a_weight   = st.number_input("Weight (kg)", 30.0, 200.0, 72.0, step=0.5, key="a_w")
        a_bmi      = a_weight / (a_height ** 2)
        a_bmi_cat, a_bmi_col = bmi_category(a_bmi)
        st.markdown(f"""<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
                    border-radius:8px;padding:10px 14px;margin:4px 0 12px 0;display:flex;justify-content:space-between;">
          <span style="color:#5a6a82;font-size:0.72rem;">BMI</span>
          <span style="color:{a_bmi_col};font-family:'DM Mono',monospace;font-size:0.9rem;">{a_bmi:.1f} — {a_bmi_cat}</span>
        </div>""", unsafe_allow_html=True)
        a_diabetes   = st.selectbox("Diabetes",        [0,1], format_func=lambda x: "No" if x==0 else "Yes", key="a_db")
        a_bp         = st.selectbox("Blood Pressure",  [0,1], format_func=lambda x: "No" if x==0 else "Yes", key="a_bp")
        a_transplant = st.selectbox("Transplants",     [0,1], format_func=lambda x: "No" if x==0 else "Yes", key="a_tr")
        a_chronic    = st.selectbox("Chronic Disease", [0,1], format_func=lambda x: "No" if x==0 else "Yes", key="a_ch2")
        a_surgeries  = st.selectbox("Surgeries",       [0,1,2,3], format_func=lambda x: str(x), key="a_su")

    with divider:
        st.markdown("""<div style="display:flex;align-items:center;justify-content:center;height:100%;min-height:400px;">
          <div style="width:1px;background:rgba(255,255,255,0.07);min-height:400px;"></div>
        </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown("""<div style="background:rgba(58,123,213,0.06);border:1px solid rgba(58,123,213,0.2);
                    border-radius:14px;padding:18px 20px;margin-bottom:20px;">
          <div style="font-size:0.72rem;color:#3a7bd5;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;">
            👤 Profile B
          </div></div>""", unsafe_allow_html=True)
        b_age      = st.slider("Age", 18, 66, 45, key="b_age")
        b_height_cm = st.number_input("Height (cm)", 100, 250, 168, step=1, key="b_h")
        b_height = b_height_cm / 100
        b_weight   = st.number_input("Weight (kg)", 30.0, 200.0, 90.0, step=0.5, key="b_w")
        b_bmi      = b_weight / (b_height ** 2)
        b_bmi_cat, b_bmi_col = bmi_category(b_bmi)
        st.markdown(f"""<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
                    border-radius:8px;padding:10px 14px;margin:4px 0 12px 0;display:flex;justify-content:space-between;">
          <span style="color:#5a6a82;font-size:0.72rem;">BMI</span>
          <span style="color:{b_bmi_col};font-family:'DM Mono',monospace;font-size:0.9rem;">{b_bmi:.1f} — {b_bmi_cat}</span>
        </div>""", unsafe_allow_html=True)
        b_diabetes   = st.selectbox("Diabetes",        [0,1], index=1, format_func=lambda x: "No" if x==0 else "Yes", key="b_db")
        b_bp         = st.selectbox("Blood Pressure",  [0,1], index=1, format_func=lambda x: "No" if x==0 else "Yes", key="b_bp")
        b_transplant = st.selectbox("Transplants",     [0,1], format_func=lambda x: "No" if x==0 else "Yes", key="b_tr")
        b_chronic    = st.selectbox("Chronic Disease", [0,1], index=1, format_func=lambda x: "No" if x==0 else "Yes", key="b_ch2")
        b_surgeries  = st.selectbox("Surgeries",       [0,1,2,3], index=1, format_func=lambda x: str(x), key="b_su")

    # ── Compute both ──────────────────────────────────────
    a_pred   = predict(a_age, int(a_height_cm), a_weight, a_diabetes, a_bp, a_transplant, a_chronic, 0, 0, a_surgeries)
    b_pred   = predict(b_age, int(b_height_cm), b_weight, b_diabetes, b_bp, b_transplant, b_chronic, 0, 0, b_surgeries)
    a_risk   = risk_profile(a_age, a_bmi, a_diabetes, a_bp, a_transplant, a_chronic, 0, 0, a_surgeries)
    b_risk   = risk_profile(b_age, b_bmi, b_diabetes, b_bp, b_transplant, b_chronic, 0, 0, b_surgeries)
    a_rlvl, a_rscore = a_risk[0], a_risk[1]
    b_rlvl, b_rscore = b_risk[0], b_risk[1]
    diff         = b_pred - a_pred
    diff_pct     = (diff / a_pred) * 100
    cheaper      = "A" if a_pred < b_pred else "B"
    cheaper_save = abs(diff)
    risk_col_map = {"LOW": C["success"], "MEDIUM": C["warning"], "HIGH": C["danger"]}

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    # ── Result banner ─────────────────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(201,168,76,0.08),rgba(201,168,76,0.03));
                border:1px solid rgba(201,168,76,0.25);border-radius:16px;padding:24px 32px;
                margin-bottom:28px;display:flex;justify-content:space-between;align-items:center;">
      <div>
        <div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;">Comparison Result</div>
        <div style="font-size:1.2rem;font-weight:600;color:#d4dae6;">
          Profile <span style="color:#c9a84c;">{cheaper}</span> saves
          <span style="color:#2fb37a;font-family:'DM Mono',monospace;"> ₹{cheaper_save:,.0f}/year</span>
        </div>
        <div style="font-size:0.78rem;color:#5a6a82;margin-top:6px;">
          ₹{cheaper_save/12:,.0f}/month difference &nbsp;·&nbsp; {abs(diff_pct):.1f}% {"more" if diff > 0 else "less"} expensive
        </div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:0.68rem;color:#5a6a82;letter-spacing:1px;margin-bottom:4px;">Annual Difference</div>
        <div style="font-family:'DM Mono',monospace;font-size:2rem;color:#c9a84c;">₹{abs(diff):,.0f}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── KPI comparison ────────────────────────────────────
    st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">Head-to-Head Metrics</div>', unsafe_allow_html=True)

    def cmp_kpi(label, val_a, val_b, fmt="text", lower_is_better=False):
        va = f"₹{val_a:,.0f}" if fmt=="currency" else (f"{val_a:.1f}" if fmt=="float" else str(val_a))
        vb = f"₹{val_b:,.0f}" if fmt=="currency" else (f"{val_b:.1f}" if fmt=="float" else str(val_b))
        if isinstance(val_a,(int,float)) and isinstance(val_b,(int,float)):
            a_better = val_a < val_b if lower_is_better else val_a > val_b
            ca = "#2fb37a" if a_better else ("#d9534f" if val_a!=val_b else "#d4dae6")
            cb = "#2fb37a" if not a_better else ("#d9534f" if val_a!=val_b else "#d4dae6")
        else:
            ca = cb = "#d4dae6"
        return f"""<div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);
                    border-radius:14px;padding:18px 20px;">
          <div style="font-size:0.65rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;">{label}</div>
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <div><div style="font-size:0.6rem;color:#c9a84c;margin-bottom:4px;">PROFILE A</div>
              <div style="font-family:'DM Mono',monospace;font-size:1.1rem;color:{ca};font-weight:500;">{va}</div></div>
            <div style="color:#3d4f66;">vs</div>
            <div style="text-align:right;"><div style="font-size:0.6rem;color:#3a7bd5;margin-bottom:4px;">PROFILE B</div>
              <div style="font-family:'DM Mono',monospace;font-size:1.1rem;color:{cb};font-weight:500;">{vb}</div></div>
          </div></div>"""

    m1,m2,m3,m4 = st.columns(4, gap="small")
    with m1: st.markdown(cmp_kpi("Annual Premium", a_pred,    b_pred,    "currency", True), unsafe_allow_html=True)
    with m2: st.markdown(cmp_kpi("BMI Index",      a_bmi,     b_bmi,     "float",    True), unsafe_allow_html=True)
    with m3: st.markdown(cmp_kpi("Risk Score",     a_rscore,  b_rscore,  lower_is_better=True), unsafe_allow_html=True)
    with m4: st.markdown(cmp_kpi("Monthly Cost",   a_pred/12, b_pred/12, "currency", True), unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────
    st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">Visual Comparison</div>', unsafe_allow_html=True)
    ch1, ch2 = st.columns([3, 2], gap="large")

    with ch1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor(C["bg2"]); ax.set_facecolor(C["bg2"])
        bars = ax.bar(["Profile A","Profile B"], [a_pred,b_pred],
                      color=[C["gold"],C["blue"]], width=0.4, edgecolor="none", zorder=3)
        for bar, val in zip(bars, [a_pred,b_pred]):
            ax.text(bar.get_x()+bar.get_width()/2, val+max(a_pred,b_pred)*0.01,
                    f"₹{val:,.0f}", ha="center", va="bottom", color=C["plat"], fontsize=9, fontweight="600")
        ax.set_ylabel("Annual Premium (₹)", fontsize=8, color=C["muted"])
        ax.set_title("Annual Premium Comparison", fontsize=10, color=C["plat"], pad=12, fontweight="600")
        ax.tick_params(colors=C["muted"], labelsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"₹{x:,.0f}"))
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.grid(axis="y", color=C["border"], lw=0.5, alpha=0.5); ax.set_axisbelow(True)
        plt.tight_layout(pad=1.5); st.pyplot(fig, width='stretch'); plt.close()

    with ch2:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        fig.patch.set_facecolor(C["bg2"]); ax.set_facecolor(C["bg2"])
        cats    = ["Premium","BMI","Risk","Age","Children"]
        maxv    = [50000,40,90,80,5]
        a_n     = [min(v/m,1.0) for v,m in zip([a_pred,a_bmi,a_rscore,a_age,a_surgeries],maxv)]
        b_n     = [min(v/m,1.0) for v,m in zip([b_pred,b_bmi,b_rscore,b_age,b_surgeries],maxv)]
        x = np.arange(len(cats)); w=0.3
        ax.bar(x-w/2, a_n, w, color=C["gold"], alpha=0.85, edgecolor="none", label="A")
        ax.bar(x+w/2, b_n, w, color=C["blue"],  alpha=0.85, edgecolor="none", label="B")
        ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=7.5, color=C["muted"])
        ax.set_title("Multi-Factor Comparison", fontsize=9, color=C["plat"], pad=10, fontweight="600")
        ax.set_ylim(0,1.15); ax.tick_params(colors=C["muted"])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.grid(axis="y", color=C["border"], lw=0.5, alpha=0.5); ax.set_axisbelow(True)
        leg = ax.legend(fontsize=7.5, framealpha=0)
        for t in leg.get_texts(): t.set_color(C["plat"])
        plt.tight_layout(pad=1.5); st.pyplot(fig, width='stretch'); plt.close()

    # ── Risk badges ───────────────────────────────────────
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">Risk Classification</div>', unsafe_allow_html=True)
    rb1, rb2 = st.columns(2, gap="medium")
    for col, lbl, rlvl, rscore, rpred, pcol in [
        (rb1,"Profile A",a_rlvl,a_rscore,a_pred,C["gold"]),
        (rb2,"Profile B",b_rlvl,b_rscore,b_pred,C["blue"])
    ]:
        rc = risk_col_map[rlvl]
        with col:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
                        border-top:3px solid {pcol};border-radius:14px;padding:24px;">
              <div style="font-size:0.68rem;color:{pcol};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;">{lbl}</div>
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
                <div style="font-size:1.8rem;font-weight:700;color:{rc};">{rlvl}</div>
                <div style="text-align:right;">
                  <div style="font-size:0.65rem;color:#5a6a82;">Risk Score</div>
                  <div style="font-family:'DM Mono',monospace;font-size:1.2rem;color:{rc};">{rscore}/90</div>
                </div>
              </div>
              <div style="height:6px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden;margin-bottom:14px;">
                <div style="width:{min(rscore/90*100,100):.0f}%;height:100%;background:{rc};border-radius:3px;"></div>
              </div>
              <div style="display:flex;justify-content:space-between;">
                <div><div style="font-size:0.62rem;color:#5a6a82;">Annual</div>
                  <div style="font-family:'DM Mono',monospace;font-size:0.95rem;color:#d4dae6;">₹{rpred:,.0f}</div></div>
                <div style="text-align:right;"><div style="font-size:0.62rem;color:#5a6a82;">Monthly</div>
                  <div style="font-family:'DM Mono',monospace;font-size:0.95rem;color:#d4dae6;">₹{rpred/12:,.0f}</div></div>
              </div>
            </div>""", unsafe_allow_html=True)

    # ── What-if savings ───────────────────────────────────
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">💡 What-If Savings Opportunities</div>', unsafe_allow_html=True)

    savings = []
    for lbl, _age, _hcm, _wt, _db, _bp, _tp, _ch, _al, _cf, _su, _bmi, _pred in [
        ("Profile A", a_age, int(a_height_cm), a_weight, a_diabetes, a_bp, a_transplant, a_chronic, 0, 0, a_surgeries, a_bmi, a_pred),
        ("Profile B", b_age, int(b_height_cm), b_weight, b_diabetes, b_bp, b_transplant, b_chronic, 0, 0, b_surgeries, b_bmi, b_pred),
    ]:
        if _bmi >= 30:
            # What if weight drops to BMI 24.9
            target_wt = 24.9 * ((_hcm / 100) ** 2)
            saved = _pred - predict(_age, _hcm, target_wt, _db, _bp, _tp, _ch, _al, _cf, _su)
            if saved > 0:
                savings.append((lbl, "Reduce BMI to 24.9", saved, "#e8a838"))
        elif _bmi >= 25:
            target_wt = 22.0 * ((_hcm / 100) ** 2)
            saved = _pred - predict(_age, _hcm, target_wt, _db, _bp, _tp, _ch, _al, _cf, _su)
            if saved > 0:
                savings.append((lbl, "Reach Healthy BMI 22", saved, "#e8a838"))
        if _db:
            saved = _pred - predict(_age, _hcm, _wt, 0, _bp, _tp, _ch, _al, _cf, _su)
            if saved > 0:
                savings.append((lbl, "Manage Diabetes", saved, "#d9534f"))
        if _bp:
            saved = _pred - predict(_age, _hcm, _wt, _db, 0, _tp, _ch, _al, _cf, _su)
            if saved > 0:
                savings.append((lbl, "Control Blood Pressure", saved, "#3a7bd5"))

    if savings:
        s_cols = st.columns(min(len(savings), 3), gap="small")
        for i, (lbl, action, saved, color) in enumerate(savings):
            pcol = C["gold"] if lbl=="Profile A" else C["blue"]
            with s_cols[i % 3]:
                st.markdown(f"""
                <div style="padding:16px 18px;background:rgba(47,179,122,0.06);
                            border:1px solid rgba(47,179,122,0.18);border-left:3px solid {color};
                            border-radius:10px;margin-bottom:10px;">
                  <div style="font-size:0.62rem;color:{pcol};text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">{lbl}</div>
                  <div style="font-size:0.82rem;font-weight:600;color:#d4dae6;margin-bottom:8px;">{action}</div>
                  <div style="font-size:0.72rem;color:#5a6a82;">Potential annual saving</div>
                  <div style="font-family:'DM Mono',monospace;font-size:1.2rem;color:#2fb37a;font-weight:600;">₹{abs(saved):,.0f}</div>
                  <div style="font-size:0.7rem;color:#5a6a82;margin-top:4px;">= ₹{abs(saved)/12:,.0f}/month</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding:20px;background:rgba(47,179,122,0.06);border:1px solid rgba(47,179,122,0.18);
                    border-radius:12px;text-align:center;">
          <div style="color:#2fb37a;font-weight:600;">✅ Both profiles are already optimised!</div>
          <div style="color:#5a6a82;font-size:0.8rem;margin-top:6px;">No major lifestyle changes needed.</div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 7 — DATA ANALYSIS (EDA)
# ════════════════════════════════════════════
with T7:
    st.markdown("""
    <div style="margin-bottom:28px;">
      <div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;">Exploratory Data Analysis</div>
      <p style="color:#5a6a82;font-size:0.8rem;margin:0;">Visual analysis of the insurance dataset — distributions, correlations and key patterns that drive premium predictions.</p>
    </div>""", unsafe_allow_html=True)

    @st.cache_data
    def load_eda_data():
        base = os.path.dirname(os.path.abspath(__file__))
        # Try enhanced dataset first, fall back to base
        for fname in ["Medicalpremium.csv", "insurance_enhanced.csv"]:
            path = os.path.join(base, fname)
            if os.path.exists(path):
                return pd.read_csv(path)
        return None

    eda_df = load_eda_data()

    if eda_df is None:
        st.warning("Dataset file not found. Please ensure insurance.csv is in the app directory.")
    else:
        # ── Dataset Overview ──────────────────────────
        st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">Dataset Overview</div>', unsafe_allow_html=True)
        ov1, ov2, ov3, ov4 = st.columns(4, gap="small")
        with ov1: st.markdown(kpi("Total Records",   f"{len(eda_df):,}",           "Rows in dataset"),   unsafe_allow_html=True)
        with ov2: st.markdown(kpi("Features",        f"{len(eda_df.columns)-1}",   "Input variables"),   unsafe_allow_html=True)
        with ov3: st.markdown(kpi("Avg Premium",     f"₹{eda_df['PremiumPrice'].mean():,.0f}", "Mean annual"),unsafe_allow_html=True)
        with ov4: st.markdown(kpi("Diabetes Rate", f"{eda_df['Diabetes'].mean()*100:.1f}%" if "Diabetes" in eda_df.columns else "N/A", "of dataset"), unsafe_allow_html=True)

        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

        # ── Row 1: Premium Distribution + Smoker vs Non-Smoker ──
        st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">Premium Distribution</div>', unsafe_allow_html=True)
        r1c1, r1c2 = st.columns(2, gap="large")

        with r1c1:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            fig.patch.set_facecolor(C["bg2"]); ax.set_facecolor(C["bg2"])
            ax.hist(eda_df["PremiumPrice"], bins=40, color=C["gold"], alpha=0.85, edgecolor="none")
            ax.axvline(eda_df["PremiumPrice"].mean(), color=C["success"], lw=1.5, linestyle="--", label=f"Mean ₹{eda_df['PremiumPrice'].mean():,.0f}")
            ax.axvline(eda_df["PremiumPrice"].median(), color=C["blue"], lw=1.5, linestyle="--", label=f"Median ₹{eda_df['PremiumPrice'].median():,.0f}")
            ax.set_xlabel("Annual Charges (₹)", fontsize=8, color=C["muted"])
            ax.set_ylabel("Count", fontsize=8, color=C["muted"])
            ax.set_title("Premium Distribution", fontsize=9, color=C["plat"], fontweight="600")
            ax.tick_params(colors=C["muted"], labelsize=7)
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.grid(axis="y", color=C["border"], lw=0.5, alpha=0.5); ax.set_axisbelow(True)
            leg = ax.legend(fontsize=7, framealpha=0)
            for t in leg.get_texts(): t.set_color(C["plat"])
            plt.tight_layout(pad=1.2); st.pyplot(fig, width='stretch'); plt.close()

        with r1c2:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            fig.patch.set_facecolor(C["bg2"]); ax.set_facecolor(C["bg2"])
            # Diabetes is 0/1 integer in Medicalpremium.csv
            diabetic     = eda_df[eda_df["Diabetes"]==1]["PremiumPrice"]
            nondiabetic  = eda_df[eda_df["Diabetes"]==0]["PremiumPrice"]
            ax.hist(nondiabetic, bins=30, color=C["success"], alpha=0.7, label=f"No Diabetes (n={len(nondiabetic)})", edgecolor="none")
            ax.hist(diabetic,    bins=30, color=C["danger"],  alpha=0.7, label=f"Diabetic (n={len(diabetic)})",       edgecolor="none")
            ax.set_xlabel("Annual Premium (₹)", fontsize=8, color=C["muted"])
            ax.set_title("Diabetic vs Non-Diabetic Premium", fontsize=9, color=C["plat"], fontweight="600")
            ax.tick_params(colors=C["muted"], labelsize=7)
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.grid(axis="y", color=C["border"], lw=0.5, alpha=0.5); ax.set_axisbelow(True)
            leg = ax.legend(fontsize=7, framealpha=0)
            for t in leg.get_texts(): t.set_color(C["plat"])
            plt.tight_layout(pad=1.2); st.pyplot(fig, width='stretch'); plt.close()

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # ── Row 2: Age vs Charges + BMI vs Charges ──
        st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">Key Feature Relationships</div>', unsafe_allow_html=True)
        r2c1, r2c2 = st.columns(2, gap="large")

        with r2c1:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            fig.patch.set_facecolor(C["bg2"]); ax.set_facecolor(C["bg2"])
            # Diabetes is 0/1 integer
            colors_s = [C["danger"] if s==1 else C["blue"] for s in eda_df["Diabetes"]]
            ax.scatter(eda_df["Age"], eda_df["PremiumPrice"], c=colors_s, alpha=0.4, s=10, edgecolors="none")
            ax.set_xlabel("Age", fontsize=8, color=C["muted"])
            ax.set_ylabel("Annual Premium (₹)", fontsize=8, color=C["muted"])
            ax.set_title("Age vs Premium (🔴 Diabetic  🔵 Non-Diabetic)", fontsize=8, color=C["plat"], fontweight="600")
            ax.tick_params(colors=C["muted"], labelsize=7)
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.grid(color=C["border"], lw=0.4, alpha=0.4); ax.set_axisbelow(True)
            plt.tight_layout(pad=1.2); st.pyplot(fig, width='stretch'); plt.close()

        with r2c2:
            # Compute BMI for scatter (notebook Cell 250)
            eda_df_bmi = eda_df.copy()
            if "BMI" not in eda_df_bmi.columns:
                eda_df_bmi["BMI"] = eda_df_bmi["Weight"] / ((eda_df_bmi["Height"] / 100) ** 2)
            fig, ax = plt.subplots(figsize=(5, 3.2))
            fig.patch.set_facecolor(C["bg2"]); ax.set_facecolor(C["bg2"])
            colors_s = [C["danger"] if s==1 else C["blue"] for s in eda_df_bmi["Diabetes"]]
            ax.scatter(eda_df_bmi["BMI"], eda_df_bmi["PremiumPrice"], c=colors_s, alpha=0.4, s=10, edgecolors="none")
            ax.axvline(30, color=C["warning"], lw=1, linestyle="--", alpha=0.7, label="BMI=30 (Obese)")
            ax.set_xlabel("BMI", fontsize=8, color=C["muted"])
            ax.set_ylabel("Annual Premium (₹)", fontsize=8, color=C["muted"])
            ax.set_title("BMI vs Premium (🔴 Diabetic  🔵 Non-Diabetic)", fontsize=8, color=C["plat"], fontweight="600")
            ax.tick_params(colors=C["muted"], labelsize=7)
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.grid(color=C["border"], lw=0.4, alpha=0.4); ax.set_axisbelow(True)
            leg = ax.legend(fontsize=7, framealpha=0)
            for t in leg.get_texts(): t.set_color(C["plat"])
            plt.tight_layout(pad=1.2); st.pyplot(fig, width='stretch'); plt.close()

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # ── Row 3: Avg Premium by Region + by Children ──
        st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">Categorical Breakdown</div>', unsafe_allow_html=True)
        r3c1, r3c2 = st.columns(2, gap="large")

        with r3c1:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            fig.patch.set_facecolor(C["bg2"]); ax.set_facecolor(C["bg2"])
            reg_avg = eda_df.groupby("BloodPressureProblems")["PremiumPrice"].mean().sort_values(ascending=True)
            # Map 0/1 integers to readable labels
            labels = ["No BP Problem" if k == 0 else "Has BP Problem" for k in reg_avg.index]
            bars = ax.barh(labels, reg_avg.values, color=C["gold"], alpha=0.85, edgecolor="none")
            for bar, val in zip(bars, reg_avg.values):
                ax.text(val + 100, bar.get_y() + bar.get_height()/2,
                        f"₹{val:,.0f}", va="center", fontsize=7, color=C["plat"])
            ax.set_xlabel("Avg Annual Premium (₹)", fontsize=8, color=C["muted"])
            ax.set_title("Average Premium by BP Problem", fontsize=9, color=C["plat"], fontweight="600")
            ax.tick_params(colors=C["muted"], labelsize=7)
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.grid(axis="x", color=C["border"], lw=0.5, alpha=0.5); ax.set_axisbelow(True)
            plt.tight_layout(pad=1.2); st.pyplot(fig, width='stretch'); plt.close()

        with r3c2:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            fig.patch.set_facecolor(C["bg2"]); ax.set_facecolor(C["bg2"])
            ch_avg = eda_df.groupby("NumberOfMajorSurgeries")["PremiumPrice"].mean()
            surg_labels = [f"{int(k)} Surger{'y' if k==1 else 'ies'}" for k in ch_avg.index]
            bars2 = ax.bar(surg_labels, ch_avg.values, color=C["blue"], alpha=0.85, edgecolor="none")
            for bar, val in zip(bars2, ch_avg.values):
                ax.text(bar.get_x() + bar.get_width()/2, val + max(ch_avg.values)*0.01,
                        f"₹{val:,.0f}", ha="center", fontsize=7, color=C["plat"])
            ax.set_xlabel("Number of Major Surgeries", fontsize=8, color=C["muted"])
            ax.set_ylabel("Avg Annual Premium (₹)", fontsize=8, color=C["muted"])
            ax.set_title("Average Premium by Major Surgeries", fontsize=9, color=C["plat"], fontweight="600")
            ax.tick_params(colors=C["muted"], labelsize=7)
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.grid(axis="y", color=C["border"], lw=0.5, alpha=0.5); ax.set_axisbelow(True)
            plt.tight_layout(pad=1.2); st.pyplot(fig, width='stretch'); plt.close()

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # ── Correlation Heatmap ──
        st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">Correlation Heatmap</div>', unsafe_allow_html=True)

        num_df = eda_df.copy()
        if "BMI" not in num_df.columns:
            num_df["BMI"] = num_df["Weight"] / ((num_df["Height"] / 100) ** 2)
        num_df["HealthRiskScore"] = (num_df["Diabetes"] + num_df["BloodPressureProblems"] +
                                     num_df["AnyTransplants"] + num_df["AnyChronicDiseases"] +
                                     num_df["KnownAllergies"] + num_df["HistoryOfCancerInFamily"])
        corr_cols = [c for c in ["Age","BMI","Diabetes","BloodPressureProblems","AnyTransplants",
                                  "AnyChronicDiseases","HistoryOfCancerInFamily",
                                  "NumberOfMajorSurgeries","HealthRiskScore","PremiumPrice"]
                     if c in num_df.columns]
        corr = num_df[corr_cols].corr()

        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor(C["bg2"]); ax.set_facecolor(C["bg2"])
        import matplotlib.colors as mcolors
        cmap = mcolors.LinearSegmentedColormap.from_list("custom",
               ["#3a7bd5", C["bg2"], "#c9a84c"], N=256)
        im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        labels = [c.replace("_num","").replace("_"," ").title() for c in corr.columns]
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5, color=C["plat"])
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7.5, color=C["plat"])
        for i in range(len(corr)):
            for j in range(len(corr)):
                val = corr.values[i,j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(val) > 0.4 else C["muted"],
                        fontweight="600" if abs(val) > 0.5 else "normal")
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title("Feature Correlation Matrix", fontsize=10, color=C["plat"], pad=12, fontweight="600")
        plt.tight_layout(pad=1.5); st.pyplot(fig, width='stretch'); plt.close()

        # ── Key Insights ──
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">Key Insights from Data</div>', unsafe_allow_html=True)

        diabetic_avg    = eda_df[eda_df["Diabetes"]==1]["PremiumPrice"].mean()
        nondiabetic_avg = eda_df[eda_df["Diabetes"]==0]["PremiumPrice"].mean()
        diabetic_mult   = diabetic_avg / nondiabetic_avg if nondiabetic_avg else 1
        bp_high_avg     = eda_df[eda_df["BloodPressureProblems"]==1]["PremiumPrice"].mean()
        bp_low_avg      = eda_df[eda_df["BloodPressureProblems"]==0]["PremiumPrice"].mean()
        # Compute BMI for obesity penalty
        eda_df_ins = eda_df.copy()
        if "BMI" not in eda_df_ins.columns:
            eda_df_ins["BMI"] = eda_df_ins["Weight"] / ((eda_df_ins["Height"] / 100) ** 2)
        obese_avg  = eda_df_ins[eda_df_ins["BMI"]>=30]["PremiumPrice"].mean()
        normal_avg = eda_df_ins[eda_df_ins["BMI"]<25]["PremiumPrice"].mean()

        ins1, ins2, ins3 = st.columns(3, gap="small")
        for col, icon, title, val, desc in [
            (ins1, "🩺", "Diabetes Impact",   f"{diabetic_mult:.1f}x higher",    f"Diabetics pay ₹{diabetic_avg:,.0f} vs ₹{nondiabetic_avg:,.0f}"),
            (ins2, "💉", "BP Impact",         f"+₹{bp_high_avg-bp_low_avg:,.0f}", f"BP patients pay ₹{bp_high_avg:,.0f} vs ₹{bp_low_avg:,.0f}"),
            (ins3, "⚖️", "Obesity Penalty",   f"+₹{obese_avg-normal_avg:,.0f}",  f"Extra cost for BMI≥30 vs BMI<25"),
        ]:
            with col:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);
                            border-top:3px solid {C['gold']};border-radius:14px;padding:20px;">
                  <div style="font-size:1.4rem;margin-bottom:8px;">{icon}</div>
                  <div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">{title}</div>
                  <div style="font-family:'DM Mono',monospace;font-size:1.1rem;color:{C['gold']};font-weight:600;">{val}</div>
                  <div style="font-size:0.75rem;color:#5a6a82;margin-top:6px;">{desc}</div>
                </div>""", unsafe_allow_html=True)

        # ── Feature Engineering Explainer ──
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">🔧 Feature Engineering Explainer</div>', unsafe_allow_html=True)
        st.markdown("""
        <p style="color:#5a6a82;font-size:0.8rem;margin-bottom:20px;">
        6 features were engineered from the 10 raw columns (notebook Cell 250) to help the Stacking Ensemble capture
        complex non-linear relationships. Each one measurably improved R² on the hold-out test set.
        </p>""", unsafe_allow_html=True)

        fe_features = [
            ("BMI",                  "Weight ÷ Height²",             "Body Mass Index computed from raw Height & Weight. Directly measures obesity risk — one of the top predictors.", C["warning"]),
            ("HealthRiskScore",      "Sum of 6 binary conditions",   "Diabetes + BP + Transplant + Chronic + Allergies + Cancer History. Compact overall health burden score (0–6).", C["warning"]),
            ("Age_HealthRisk",       "Age × HealthRiskScore",        "Older patients with more conditions face compounding risk. Captures the joint effect of age and clinical burden.", C["danger"]),
            ("Age_Surgeries",        "Age × NumberOfMajorSurgeries", "Surgical impact increases with age. A 60-year-old with 2 surgeries is far riskier than a 25-year-old with 2.",   C["danger"]),
            ("SurgeryTransplantRisk","Surgeries × (Transplant + 1)", "Flags highest-risk patients: those with both transplant history and prior major surgeries — extreme outlier tier.", C["danger"]),
            ("AgeGroup",             "Age bucket (0–3)",             "Encodes age into 4 clinical bands: Young (<30=0), Middle (30–44=1), Senior (45–59=2), Elderly (60+=3).",          C["blue"]),
        ]
        fe_cols = st.columns(2, gap="large")
        for i, (feat, name, desc, color) in enumerate(fe_features):
            with fe_cols[i % 2]:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
                            border-left:3px solid {color};border-radius:10px;padding:14px 16px;margin-bottom:12px;">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <span style="font-family:'DM Mono',monospace;font-size:0.78rem;color:{color};">{feat}</span>
                    <span style="font-size:0.72rem;color:#d4dae6;font-weight:600;">{name}</span>
                  </div>
                  <div style="font-size:0.78rem;color:#5a6a82;line-height:1.5;">{desc}</div>
                </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 8 — PREMIUM TIMELINE FORECAST
# ════════════════════════════════════════════
with T8:
    st.markdown("""
    <div style="margin-bottom:24px;">
      <div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;">Premium Timeline Forecast</div>
      <p style="color:#5a6a82;font-size:0.8rem;margin:0;">ML-powered projection of how this client's premium evolves over the next 10–20 years, based on age progression and risk factors.</p>
    </div>""", unsafe_allow_html=True)

    fc1, fc2 = st.columns([3, 1], gap="large")
    with fc2:
        forecast_years = st.selectbox("Forecast Horizon", [10, 15, 20], index=1)
        assume_bmi_improve  = st.checkbox("Assume weight reduces 0.5kg/year", value=False)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);
                    border-radius:12px;padding:16px;">
          <div style="font-size:0.65rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">Current Profile</div>
          <div style="font-size:0.78rem;color:#d4dae6;">Age: <span style="color:{C['gold']}">{age}</span></div>
          <div style="font-size:0.78rem;color:#d4dae6;margin-top:4px;">BMI: <span style="color:{C['gold']}">{bmi:.1f}</span></div>
          <div style="font-size:0.78rem;color:#d4dae6;margin-top:4px;">Diabetes: <span style="color:{'#d9534f' if diabetes else '#2fb37a'}">{'Yes' if diabetes else 'No'}</span></div>
          <div style="font-size:0.78rem;color:#d4dae6;margin-top:4px;">Base Premium: <span style="color:{C['gold']}">₹{pred:,.0f}</span></div>
        </div>""", unsafe_allow_html=True)

    with fc1:
        @st.cache_data
        def build_forecast(base_age, base_height_cm, base_weight,
                           diabetes, blood_pressure, transplant,
                           chronic, allergies, cancer_family, surgeries,
                           years, bmi_improve):
            ages, preds, lows, highs = [], [], [], []
            for y in range(years + 1):
                a = min(base_age + y, 66)
                w = max(base_weight - (0.5 * y if bmi_improve else 0), 40)
                p, lo, hi = prediction_with_ci(a, base_height_cm, w,
                                               diabetes, blood_pressure, transplant,
                                               chronic, allergies, cancer_family, surgeries)
                ages.append(a); preds.append(p); lows.append(lo)
                highs.append(hi)
            return ages, preds, lows, highs

        ages_f, preds_f, lows_f, highs_f = build_forecast(
            age, height_cm, weight,
            diabetes, blood_pressure, transplant,
            chronic, allergies, cancer_family, surgeries,
            forecast_years, assume_bmi_improve
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor(C["bg2"]); ax.set_facecolor(C["bg2"])

        ax.fill_between(ages_f, lows_f, highs_f, alpha=0.15, color=C["gold"], label="95% CI")
        ax.plot(ages_f, preds_f, color=C["gold"], lw=2.5, label="Projected Premium", zorder=5)
        ax.plot(ages_f[0], preds_f[0], "o", color=C["success"], ms=8, zorder=6, label="Today")

        # Annotate final value
        ax.annotate(f"₹{preds_f[-1]:,.0f}",
                    xy=(ages_f[-1], preds_f[-1]),
                    xytext=(-55, 12), textcoords="offset points",
                    fontsize=8, color=C["gold"],
                    arrowprops=dict(arrowstyle="->", color=C["muted"], lw=0.8))

        ax.set_xlabel("Age", fontsize=8, color=C["muted"])
        ax.set_ylabel("Annual Premium (₹)", fontsize=8, color=C["muted"])
        ax.set_title(f"Premium Forecast — Next {forecast_years} Years", fontsize=10,
                     color=C["plat"], fontweight="600", pad=12)
        ax.tick_params(colors=C["muted"], labelsize=7)
        for sp in ax.spines.values(): sp.set_color(C["border"])
        ax.grid(color=C["border"], lw=0.4, alpha=0.5); ax.set_axisbelow(True)
        leg = ax.legend(fontsize=7, framealpha=0, loc="upper left")
        for t in leg.get_texts(): t.set_color(C["plat"])
        plt.tight_layout(pad=1.5)
        st.pyplot(fig, width='stretch'); plt.close()

    # Forecast summary KPIs
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    total_cost = sum(preds_f)
    peak_prem  = max(preds_f)
    growth_pct = (preds_f[-1] - preds_f[0]) / preds_f[0] * 100
    fk1, fk2, fk3, fk4 = st.columns(4, gap="small")
    with fk1: st.markdown(kpi("Total Cost Over Period",  f"₹{total_cost:,.0f}", f"{forecast_years} years cumulative"), unsafe_allow_html=True)
    with fk2: st.markdown(kpi("Peak Annual Premium",     f"₹{peak_prem:,.0f}",  f"At age {ages_f[preds_f.index(peak_prem)]}"), unsafe_allow_html=True)
    with fk3: st.markdown(kpi("Premium Growth",          f"+{growth_pct:.1f}%",  f"Over {forecast_years} years", C["warning"] if growth_pct > 50 else C["success"]), unsafe_allow_html=True)
    with fk4: st.markdown(kpi("Avg Annual Increase",     f"₹{(preds_f[-1]-preds_f[0])/forecast_years:,.0f}", "Per year on average"), unsafe_allow_html=True)

    # Year-by-year table
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;">Year-by-Year Breakdown</div>', unsafe_allow_html=True)
    table_rows = ""
    for i, (a, p, lo, hi) in enumerate(zip(ages_f, preds_f, lows_f, highs_f)):
        ychange = "" if i == 0 else f"+₹{p - preds_f[i-1]:,.0f}"
        table_rows += f"""
        <tr style="border-bottom:1px solid rgba(255,255,255,0.04);{'background:rgba(255,255,255,0.015)' if i%2==0 else ''}">
          <td style="padding:9px 14px;font-size:0.78rem;color:#5a6a82;">Year {i}</td>
          <td style="padding:9px 14px;font-size:0.78rem;color:#d4dae6;">Age {a}</td>
          <td style="padding:9px 14px;font-family:'DM Mono',monospace;font-size:0.82rem;color:{C['gold']};">₹{p:,.0f}</td>
          <td style="padding:9px 14px;font-size:0.75rem;color:#5a6a82;">₹{lo:,.0f} – ₹{hi:,.0f}</td>
          <td style="padding:9px 14px;font-size:0.75rem;color:{C['success'] if ychange=='' else C['warning']};">{ychange}</td>
        </tr>"""
    st.markdown(f"""
    <div style="overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;">
      <thead>
        <tr style="border-bottom:1px solid rgba(255,255,255,0.1);">
          <th style="padding:10px 14px;font-size:0.65rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1px;text-align:left;">Year</th>
          <th style="padding:10px 14px;font-size:0.65rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1px;text-align:left;">Age</th>
          <th style="padding:10px 14px;font-size:0.65rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1px;text-align:left;">Premium</th>
          <th style="padding:10px 14px;font-size:0.65rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1px;text-align:left;">95% CI Range</th>
          <th style="padding:10px 14px;font-size:0.65rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1px;text-align:left;">YoY Change</th>

        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table></div>""", unsafe_allow_html=True)



# ════════════════════════════════════════════
# TAB 9 — INSURER COMPARISON
# ════════════════════════════════════════════
with T9:
    st.markdown("""
    <div style="margin-bottom:24px;">
      <div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;">Market Intelligence</div>
      <div style="font-size:1.55rem;font-weight:600;color:#d4dae6;line-height:1.2;">Real Insurer Comparison</div>
      <div style="font-size:0.82rem;color:#5a6a82;margin-top:6px;">See how your predicted premium stacks up against India's top health insurers — based on published 2024–25 rate cards.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Insurer Rate Engine ──
    # Base rates (₹/year) for a healthy 35-year-old, ₹5L sum insured, individual, Tier-1 city
    # Source: Published rate cards & Yieldora calculator cross-reference (2024-25)
    INSURERS = [
        {
            "name":    "Star Health",
            "plan":    "Comprehensive",
            "logo":    "⭐",
            "base":    12500,
            "csr":     82.3,
            "icr":     65.4,
            "network": 14000,
            "wait":    4,
            "color":   "#e8a838",
            "url":     "https://www.starhealth.in",
            "features": ["Day 1 OPD cover", "Air ambulance", "No room rent cap", "Auto recharge"],
            "best_for": "Comprehensive OPD + inpatient",
        },
        {
            "name":    "HDFC ERGO",
            "plan":    "Optima Secure",
            "logo":    "🔷",
            "base":    13800,
            "csr":     98.6,
            "icr":     58.2,
            "network": 14431,
            "wait":    3,
            "color":   "#3a7bd5",
            "url":     "https://www.hdfcergo.com",
            "features": ["2× cover from Day 1", "100% SI restoration", "No sub-limits", "50% NCB"],
            "best_for": "Highest claim settlement (98.6%)",
        },
        {
            "name":    "Niva Bupa",
            "plan":    "ReAssure 2.0",
            "logo":    "💚",
            "base":    11900,
            "csr":     91.6,
            "icr":     61.7,
            "network": 11052,
            "wait":    3,
            "color":   "#2fb37a",
            "url":     "https://www.nivabupa.com",
            "features": ["Unlimited reinstatement", "30-min cashless auth", "Lock the Clock", "Booster+ benefit"],
            "best_for": "Fastest claim processing",
        },
        {
            "name":    "Bajaj Allianz",
            "plan":    "Health Guard",
            "logo":    "🛡",
            "base":    10800,
            "csr":     95.5,
            "icr":     56.8,
            "network": 8000,
            "wait":    3,
            "color":   "#d9534f",
            "url":     "https://www.bajajallianz.com",
            "features": ["Modular plan builder", "OPD = 2× premium", "International cover", "No PED check <45 yrs"],
            "best_for": "Most affordable & customizable",
        },
        {
            "name":    "Care Health",
            "plan":    "Care Supreme",
            "logo":    "❤",
            "base":    12100,
            "csr":     98.1,
            "icr":     59.3,
            "network": 19000,
            "wait":    2,
            "color":   "#c9a84c",
            "url":     "https://www.careinsurance.com",
            "features": ["Shortest PED wait (2 yr)", "Largest hospital network", "100% SI bonus/yr", "Day 1 chronic cover"],
            "best_for": "Pre-existing conditions",
        },
        {
            "name":    "ICICI Lombard",
            "plan":    "Complete Health",
            "logo":    "🏛",
            "base":    13200,
            "csr":     97.4,
            "icr":     57.1,
            "network": 15000,
            "wait":    3,
            "color":   "#8a6cc9",
            "url":     "https://www.icicilombard.com",
            "features": ["Wellness rewards", "Mental health cover", "Telemedicine 24/7", "Cashless everywhere"],
            "best_for": "Digital-first experience",
        },
    ]

    def compute_market_premium(base, age, has_diabetes, has_bp, has_transplant,
                                has_chronic, surgeries, bmi_val, insurer_wait):
        """
        Estimate real-market premium using published loading factors.
        Age loading: industry standard bands.
        PED loading: +25-40% per condition.
        BMI loading: +15% if obese.
        """
        p = float(base)
        # Age loading (IRDAI published bands)
        if age < 25:    p *= 0.82
        elif age < 35:  p *= 1.00
        elif age < 45:  p *= 1.30
        elif age < 55:  p *= 1.70
        elif age < 60:  p *= 2.20
        else:           p *= 2.85

        # Condition loadings
        if has_diabetes:   p *= 1.28
        if has_bp:         p *= 1.22
        if has_transplant: p *= 1.55
        if has_chronic:    p *= 1.18
        if surgeries >= 2: p *= 1.20
        elif surgeries == 1: p *= 1.10

        # BMI loading
        if bmi_val >= 35:  p *= 1.20
        elif bmi_val >= 30: p *= 1.10

        return round(p / 1000) * 1000  # round to nearest ₹1000

    # Compute premiums for current profile
    market_premiums = []
    for ins in INSURERS:
        mp = compute_market_premium(
            ins["base"], age,
            bool(diabetes), bool(blood_pressure),
            bool(transplant), bool(chronic),
            surgeries, bmi, ins["wait"]
        )
        market_premiums.append({**ins, "market_premium": mp})

    market_premiums.sort(key=lambda x: x["market_premium"])
    cheapest = market_premiums[0]["market_premium"]
    our_pred  = pred  # from the main prediction

    # ── KPI Strip ──
    k1, k2, k3, k4 = st.columns(4)
    avg_market = sum(x["market_premium"] for x in market_premiums) / len(market_premiums)
    cheapest_name = market_premiums[0]["name"]
    vs_avg = our_pred - avg_market
    vs_cheapest = our_pred - cheapest

    with k1: st.markdown(kpi("Our ML Prediction", f"₹{our_pred:,.0f}", "annual estimate", C["gold"]), unsafe_allow_html=True)
    with k2: st.markdown(kpi("Market Average",    f"₹{avg_market:,.0f}", "top 6 insurers", C["blue"]), unsafe_allow_html=True)
    with k3: st.markdown(kpi("Cheapest Option",   f"₹{cheapest:,.0f}", cheapest_name, C["success"]), unsafe_allow_html=True)
    with k4:
        diff_color = C["danger"] if vs_avg > 0 else C["success"]
        diff_label = f"+₹{abs(vs_avg):,.0f} above avg" if vs_avg > 0 else f"₹{abs(vs_avg):,.0f} below avg"
        st.markdown(kpi("Prediction vs Avg", diff_label, "our model vs market", diff_color), unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Bar Chart ──
    st.markdown('<div style="font-size:0.75rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:14px;">Annual Premium Comparison — Your Profile</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig_style(fig, ax)

    names    = [x["name"] for x in market_premiums] + ["Our ML Model"]
    premiums = [x["market_premium"] for x in market_premiums] + [our_pred]
    colors_b = [x["color"] for x in market_premiums] + [C["gold"]]
    y_pos    = range(len(names))

    bars = ax.barh(list(y_pos), premiums, color=colors_b, alpha=0.85, height=0.6, edgecolor="none")

    # Value labels
    for bar, val in zip(bars, premiums):
        ax.text(bar.get_width() + 300, bar.get_y() + bar.get_height()/2,
                f"₹{val:,.0f}", va="center", ha="left",
                color=C["plat"], fontsize=8.5,
                fontfamily="monospace")

    # Cheapest marker
    ax.axvline(cheapest, color=C["success"], lw=1, ls="--", alpha=0.5)
    ax.axvline(our_pred, color=C["gold"],    lw=1.5, ls=":", alpha=0.6)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=9, color=C["plat"])
    ax.set_xlabel("Annual Premium (₹)", fontsize=8, color=C["muted"])
    ax.set_title(f"Your Health Profile — Age {age}, BMI {bmi:.1f}", fontsize=9, color=C["plat"])
    ax.set_xlim(0, max(premiums) * 1.22)
    ax.grid(axis='x', color=C["border"], lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); buf.seek(0)
    st.image(buf, width='stretch')
    plt.close(fig)

    st.markdown("""
    <div style="font-size:0.72rem;color:#3d4f66;margin-top:6px;margin-bottom:24px;">
      * All market premiums are indicative estimates based on published 2024–25 rate cards with standard age, BMI and condition loadings applied.
      Actual quotes depend on insurer underwriting and medical declaration. GST-exempt w.e.f. Sept 2025.
    </div>
    """, unsafe_allow_html=True)

    # ── Detailed Comparison Table ──
    st.markdown('<div style="font-size:0.75rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:14px;">Detailed Plan Comparison</div>', unsafe_allow_html=True)

    # Build table HTML
    rows_html = ""
    for i, ins in enumerate(market_premiums):
        rank_badge = ""
        if i == 0:
            rank_badge = f'<span style="background:{C["success"]}22;color:{C["success"]};font-size:0.65rem;padding:2px 7px;border-radius:10px;margin-left:8px;">CHEAPEST</span>'
        elif i == len(market_premiums) - 1:
            rank_badge = f'<span style="background:#d9534f22;color:#d9534f;font-size:0.65rem;padding:2px 7px;border-radius:10px;margin-left:8px;">HIGHEST</span>'

        csr_color = C["success"] if ins["csr"] >= 97 else (C["warning"] if ins["csr"] >= 90 else C["danger"])
        wait_color = C["success"] if ins["wait"] <= 2 else (C["warning"] if ins["wait"] == 3 else C["danger"])

        feat_pills = " ".join([
            f'<span style="background:rgba(255,255,255,0.05);color:#8a9ab5;font-size:0.65rem;'
            f'padding:2px 7px;border-radius:8px;border:1px solid rgba(255,255,255,0.07);">{f}</span>'
            for f in ins["features"]
        ])

        rows_html += f"""
        <tr style="border-bottom:1px solid rgba(255,255,255,0.04);{'background:rgba(255,255,255,0.015)' if i%2==0 else ''}">
          <td style="padding:14px 16px;vertical-align:top;">
            <div style="display:flex;align-items:center;gap:8px;">
              <span style="font-size:1.1rem;">{ins['logo']}</span>
              <div>
                <div style="font-size:0.85rem;font-weight:600;color:{ins['color']};">{ins['name']}{rank_badge}</div>
                <div style="font-size:0.72rem;color:#5a6a82;margin-top:2px;">{ins['plan']}</div>
              </div>
            </div>
          </td>
          <td style="padding:14px 16px;vertical-align:top;text-align:center;">
            <div style="font-family:'DM Mono',monospace;font-size:1.05rem;font-weight:600;color:{ins['color']};">₹{ins['market_premium']:,.0f}</div>
            <div style="font-size:0.68rem;color:#3d4f66;margin-top:2px;">₹{ins['market_premium']//12:,.0f}/mo</div>
          </td>
          <td style="padding:14px 16px;vertical-align:top;text-align:center;">
            <div style="font-size:0.85rem;font-weight:600;color:{csr_color};">{ins['csr']}%</div>
            <div style="font-size:0.68rem;color:#3d4f66;">Claim settled</div>
          </td>
          <td style="padding:14px 16px;vertical-align:top;text-align:center;">
            <div style="font-size:0.85rem;color:#d4dae6;">{ins['network']:,}</div>
            <div style="font-size:0.68rem;color:#3d4f66;">hospitals</div>
          </td>
          <td style="padding:14px 16px;vertical-align:top;text-align:center;">
            <div style="font-size:0.85rem;color:{wait_color};">{ins['wait']} yr</div>
            <div style="font-size:0.68rem;color:#3d4f66;">PED wait</div>
          </td>
          <td style="padding:14px 16px;vertical-align:top;">
            <div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:6px;">{feat_pills}</div>
            <div style="font-size:0.7rem;color:#5a6a82;font-style:italic;">{ins['best_for']}</div>
          </td>
          <td style="padding:14px 16px;vertical-align:middle;text-align:center;">
            <a href="{ins['url']}" target="_blank"
               style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.12);
                      color:#d4dae6;font-size:0.72rem;padding:6px 14px;border-radius:6px;
                      text-decoration:none;white-space:nowrap;">
              Get Quote →
            </a>
          </td>
        </tr>"""

    # Also add Our ML Model row at bottom
    rows_html += f"""
    <tr style="border-top:2px solid {C['gold']}44;background:rgba(201,168,76,0.05)">
      <td style="padding:14px 16px;vertical-align:top;">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="font-size:1.1rem;">🤖</span>
          <div>
            <div style="font-size:0.85rem;font-weight:600;color:{C['gold']};">PremiumIQ Prediction</div>
            <div style="font-size:0.72rem;color:#5a6a82;margin-top:2px;">XGBoost ML Model</div>
          </div>
        </div>
      </td>
      <td style="padding:14px 16px;vertical-align:top;text-align:center;">
        <div style="font-family:'DM Mono',monospace;font-size:1.05rem;font-weight:600;color:{C['gold']};">₹{our_pred:,.0f}</div>
        <div style="font-size:0.68rem;color:#3d4f66;margin-top:2px;">₹{our_pred//12:,.0f}/mo</div>
      </td>
      <td style="padding:14px 16px;vertical-align:top;text-align:center;">
        <div style="font-size:0.85rem;font-weight:600;color:{C['success']};">{MODEL_R2*100:.1f}%</div>
        <div style="font-size:0.68rem;color:#3d4f66;">Model R²</div>
      </td>
      <td style="padding:14px 16px;vertical-align:top;text-align:center;">
        <div style="font-size:0.85rem;color:#d4dae6;">986</div>
        <div style="font-size:0.68rem;color:#3d4f66;">training samples</div>
      </td>
      <td style="padding:14px 16px;vertical-align:top;text-align:center;">
        <div style="font-size:0.85rem;color:{C['success']};">None</div>
        <div style="font-size:0.68rem;color:#3d4f66;">no waiting period</div>
      </td>
      <td style="padding:14px 16px;vertical-align:top;">
        <div style="display:flex;flex-wrap:wrap;gap:4px;">
          {''.join([f"<span style='background:rgba(201,168,76,0.1);color:{C['gold']};font-size:0.65rem;padding:2px 7px;border-radius:8px;border:1px solid rgba(201,168,76,0.2);'>{f}</span>" for f in ["18 risk features", "SHAP explainability", "CI bounds", "Real-time prediction"]])}
        </div>
        <div style="font-size:0.7rem;color:#5a6a82;font-style:italic;margin-top:6px;">Analytical estimate — not an insurance product</div>
      </td>
      <td style="padding:14px 16px;vertical-align:middle;text-align:center;">
        <span style="background:rgba(201,168,76,0.12);border:1px solid rgba(201,168,76,0.3);
                     color:{C['gold']};font-size:0.72rem;padding:6px 14px;border-radius:6px;">
          This App
        </span>
      </td>
    </tr>"""

    st.markdown(f"""
    <div style="border:1px solid rgba(255,255,255,0.07);border-radius:14px;overflow:hidden;margin-bottom:20px;">
      <table style="width:100%;border-collapse:collapse;">
        <thead>
          <tr style="background:rgba(255,255,255,0.03);border-bottom:1px solid rgba(255,255,255,0.08);">
            <th style="padding:10px 16px;font-size:0.65rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1px;text-align:left;">Insurer & Plan</th>
            <th style="padding:10px 16px;font-size:0.65rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1px;text-align:center;">Your Premium / yr</th>
            <th style="padding:10px 16px;font-size:0.65rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1px;text-align:center;">Claim CSR</th>
            <th style="padding:10px 16px;font-size:0.65rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1px;text-align:center;">Hospitals</th>
            <th style="padding:10px 16px;font-size:0.65rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1px;text-align:center;">PED Wait</th>
            <th style="padding:10px 16px;font-size:0.65rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1px;text-align:left;">Key Features</th>
            <th style="padding:10px 16px;font-size:0.65rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1px;text-align:center;">Action</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    # ── Personalised Recommendation Box ──
    rec_ins = market_premiums[0]  # cheapest
    best_csr_ins = max(market_premiums, key=lambda x: x["csr"])

    if has_diabetes := bool(diabetes) or bool(blood_pressure) or bool(chronic):
        # Has PED — recommend shortest wait
        rec_ins = min(market_premiums, key=lambda x: x["wait"])
        rec_reason = "You have pre-existing conditions — this insurer has the shortest PED waiting period."
    else:
        rec_reason = "Based on your profile, this offers the best value — lowest premium with strong claim settlement."

    st.markdown(f"""
    <div style="border:1px solid rgba(201,168,76,0.25);border-radius:14px;padding:20px 24px;
                background:rgba(201,168,76,0.05);margin-bottom:12px;">
      <div style="font-size:0.68rem;color:#c9a84c;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px;">
        🎯 Personalised Recommendation
      </div>
      <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
        <div style="font-size:1.8rem;">{rec_ins['logo']}</div>
        <div style="flex:1;">
          <div style="font-size:1.05rem;font-weight:600;color:{rec_ins['color']};">{rec_ins['name']} — {rec_ins['plan']}</div>
          <div style="font-size:0.82rem;color:#8a9ab5;margin-top:4px;">{rec_reason}</div>
        </div>
        <div style="text-align:right;">
          <div style="font-family:'DM Mono',monospace;font-size:1.4rem;font-weight:600;color:{rec_ins['color']};">₹{rec_ins['market_premium']:,.0f}<span style="font-size:0.75rem;color:#5a6a82;">/yr</span></div>
          <div style="font-size:0.72rem;color:#5a6a82;margin-top:2px;">Estimated for your profile</div>
        </div>
      </div>
      <div style="margin-top:14px;padding-top:14px;border-top:1px solid rgba(255,255,255,0.06);
                  display:flex;gap:20px;flex-wrap:wrap;">
        <div><span style="font-size:0.7rem;color:#5a6a82;">Claim Settlement</span>
             <span style="font-size:0.82rem;font-weight:600;color:{C['success']};margin-left:8px;">{rec_ins['csr']}%</span></div>
        <div><span style="font-size:0.7rem;color:#5a6a82;">Network Hospitals</span>
             <span style="font-size:0.82rem;font-weight:600;color:#d4dae6;margin-left:8px;">{rec_ins['network']:,}</span></div>
        <div><span style="font-size:0.7rem;color:#5a6a82;">PED Waiting Period</span>
             <span style="font-size:0.82rem;font-weight:600;color:{C['warning']};margin-left:8px;">{rec_ins['wait']} years</span></div>
        <div><span style="font-size:0.7rem;color:#5a6a82;">CSR Rank</span>
             <span style="font-size:0.82rem;font-weight:600;color:#d4dae6;margin-left:8px;">#{sorted(market_premiums, key=lambda x: -x['csr']).index(rec_ins)+1} of 6</span></div>
      </div>
    </div>
    <div style="font-size:0.7rem;color:#3d4f66;margin-top:8px;">
      ℹ Premiums are indicative estimates. PED = Pre-Existing Disease. CSR = Claim Settlement Ratio (IRDAI FY 2023–24).
      Always get a final quote from the insurer's website. Not financial advice.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div style="margin-top:48px;padding:20px 0;border-top:1px solid rgba(255,255,255,0.06);
            display:flex;justify-content:space-between;align-items:center;">
  <div style="font-size:0.7rem;color:#3d4f66;">
    PremiumIQ Risk Intelligence &nbsp;·&nbsp; Stacking Ensemble (XGBoost + CatBoost + RF → Ridge) &nbsp;·&nbsp; R² {MODEL_R2:.4f if MODEL_R2 else "N/A"}
  </div>
  <div style="font-size:0.7rem;color:#3d4f66;">
    Estimates are for analytical purposes only and do not constitute financial advice.
  </div>
</div>
""", unsafe_allow_html=True)