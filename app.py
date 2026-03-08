import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
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
                                    Table, TableStyle, HRFlowable)
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
    initial_sidebar_state="expanded",
    menu_items={}
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

/* Hide footer, hamburger menu and toolbar */
#MainMenu        { display: none !important; }
footer           { display: none !important; }
header           { visibility: hidden !important; }
[data-testid="stToolbar"]      { display: none !important; }
[data-testid="stDecoration"]   { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }

/* Hide keyboard shortcut tooltip only — not dropdowns */
[data-testid="stToolbar"]      { display: none !important; }
[data-testid="stDecoration"]   { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
iframe[title="streamlit_shortcuts"] { display: none !important; }

/* Fix dropdown options visibility */
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

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--navy-2); }
::-webkit-scrollbar-thumb { background: var(--muted2); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base    = os.path.dirname(os.path.abspath(__file__))
    model   = joblib.load(os.path.join(base, "model.pkl"))
    columns = joblib.load(os.path.join(base, "columns.pkl"))
    scaler  = joblib.load(os.path.join(base, "scaler.pkl")) if os.path.exists(os.path.join(base, "scaler.pkl")) else None
    return model, columns, scaler

@st.cache_resource
def load_shap_explainer(_model, _columns, _scaler):
    """
    SHAP explainer compatible with XGBoost 3.2.0.
    Uses shap.Explainer (new API) instead of TreeExplainer.
    """
    try:
        np.random.seed(42)
        n = 200
        bg_raw = pd.DataFrame({
            "age"     : np.random.randint(18, 65, n).astype(float),
            "bmi"     : np.round(np.random.normal(30.5, 6.5, n).clip(15, 53), 1),
            "children": np.random.choice([0,1,2,3,4,5], n,
                                          p=[0.43,0.24,0.18,0.10,0.03,0.02]).astype(float),
            "sex"     : np.random.choice(["male","female"], n),
            "smoker"  : np.random.choice(["yes","no"], n, p=[0.20,0.80]),
            "region"  : np.random.choice(
                            ["northeast","northwest","southeast","southwest"], n)
        })
        bg_enc = pd.get_dummies(bg_raw, drop_first=True)
        for c in _columns:
            if c not in bg_enc.columns:
                bg_enc[c] = 0
        cols   = [str(c) for c in _columns]
        bg_enc = bg_enc[cols].astype(np.float64)
        bg_enc.columns = cols

        if _scaler is not None:
            X_bg = pd.DataFrame(_scaler.transform(bg_enc), columns=cols)
        else:
            X_bg = bg_enc.copy()

        # ── Use shap.Explainer (XGBoost 3.x compatible) ───────────────────
        # Pass model.predict so SHAP treats it as a black-box —
        # avoids all internal XGBoost config parsing that causes the error
        def model_predict(data):
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data, columns=cols)
            return _model.predict(data)

        explainer = shap.Explainer(model_predict, X_bg, algorithm="permutation",
                                   max_evals=2 * len(cols) + 1)
        return explainer, X_bg

    except Exception as e:
        st.error(f"SHAP explainer error: {e}")
        return None, None

model, model_columns, scaler = load_artifacts()


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def predict(age, bmi, children, sex, smoker, region):
    d  = {"age": age, "bmi": bmi, "children": children,
          "sex": sex, "smoker": smoker, "region": region}
    df = pd.get_dummies(pd.DataFrame([d]))
    for c in model_columns:
        if c not in df.columns: df[c] = 0
    df = df[model_columns]
    if scaler: df = scaler.transform(df)
    return float(np.exp(model.predict(df)[0]))

def risk_profile(age, bmi, smoker):
    score = 0
    hi, med, lo = [], [], []
    if smoker == "yes":  score += 40; hi.append(("🚬 Active Smoker", "Highest single risk factor"))
    else:                lo.append(("✅ Non-Smoker", "Significant premium advantage"))
    if bmi >= 35:        score += 30; hi.append((f"⚠ BMI {bmi:.1f}", "Obese — high health risk"))
    elif bmi >= 30:      score += 18; med.append((f"△ BMI {bmi:.1f}", "Overweight — moderate risk"))
    elif bmi >= 25:      score += 8;  med.append((f"◇ BMI {bmi:.1f}", "Borderline — monitor weight"))
    else:                lo.append((f"✅ BMI {bmi:.1f}", "Healthy range"))
    if age >= 55:        score += 20; hi.append((f"⚠ Age {age}", "Senior bracket"))
    elif age >= 40:      score += 10; med.append((f"△ Age {age}", "Middle-aged bracket"))
    else:                lo.append((f"✅ Age {age}", "Young — lower baseline risk"))
    level = "HIGH" if score >= 50 else ("MEDIUM" if score >= 25 else "LOW")
    return level, score, hi, med, lo

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


def generate_pdf(age, height, weight, bmi, bmi_cat, children, sex, smoker, region,
                 pred, monthly, risk_lvl, risk_score, fhi, fmed, flo):
    """Generate a professional dark-theme client insurance report as PDF bytes."""
    from reportlab.platypus import KeepInFrame
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas as rl_canvas

    buf  = io.BytesIO()
    W, H = A4

    # ── Colours ──────────────────────────────────────────
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

    # ── Style helper ─────────────────────────────────────
    def S(name, **kw):
        base = dict(fontName="Helvetica", fontSize=9, textColor=PLAT,
                    leading=13, spaceAfter=0, spaceBefore=0,
                    backColor=colors.transparent)
        base.update(kw)
        return ParagraphStyle(name, **base)

    # ── Canvas background painter ─────────────────────────
    def dark_bg(canv, doc):
        canv.saveState()
        canv.setFillColor(NAVY)
        canv.rect(0, 0, W, H, fill=1, stroke=0)
        # Subtle gold gradient band at top
        canv.setFillColor(colors.HexColor("#0a1628"))
        canv.rect(0, H - 28*mm, W, 28*mm, fill=1, stroke=0)
        # Gold top border line
        canv.setStrokeColor(GOLD)
        canv.setLineWidth(2)
        canv.line(0, H, W, H)
        # Bottom border
        canv.setStrokeColor(NAVY4)
        canv.setLineWidth(0.5)
        canv.line(18*mm, 12*mm, W - 18*mm, 12*mm)
        canv.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=18*mm, bottomMargin=18*mm,
                            onFirstPage=dark_bg, onLaterPages=dark_bg)
    story = []

    # ══════════════════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════════════════
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
                      S("d1", fontSize=9, textColor=PLAT,
                        alignment=TA_RIGHT)),
            Paragraph(datetime.now().strftime("%I:%M %p"),
                      S("d2", fontSize=8, textColor=MUTED,
                        alignment=TA_RIGHT)),
            Paragraph("XGBoost  ·  R\u00b2 85.64%",
                      S("d3", fontSize=7.5, textColor=MUTED,
                        alignment=TA_RIGHT)),
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

    # ══════════════════════════════════════════════════════
    # KPI STRIP — 5 cards
    # ══════════════════════════════════════════════════════
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
        kcard("Smoker",         "Yes" if smoker=="yes" else "No",
              "+40 pts" if smoker=="yes" else "0 pts",
              DANGER if smoker=="yes" else SUCCESS),
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
        ("ROUNDEDCORNERS",[6]),
    ]))
    story.append(kpi_row)
    story.append(Spacer(1, 5*mm))

    # ══════════════════════════════════════════════════════
    # SECTION HEADER HELPER
    # ══════════════════════════════════════════════════════
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
            ("ROUNDEDCORNERS", [4]),
        ]))
        return t

    # ══════════════════════════════════════════════════════
    # TWO COLUMN BODY
    # ══════════════════════════════════════════════════════
    region_fmt = (region.replace("north","North ").replace("south","South ")
                        .replace("east","East").replace("west","West").title())

    # ── LEFT COLUMN ───────────────────────────────────────
    left = []
    left += sec("Client Profile")
    left.append(row_tbl([
        ["Age",            f"{age} years"],
        ["Height",         f"{height:.2f} m"],
        ["Weight",         f"{weight:.1f} kg"],
        ["BMI",            f"{bmi:.1f} ({bmi_cat})"],
        ["Dependents",     str(children)],
        ["Sex",            sex.title()],
        ["Smoking",        "Active Smoker" if smoker=="yes" else "Non-Smoker"],
        ["Region",         region_fmt],
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

    # ── RIGHT COLUMN ──────────────────────────────────────
    right = []
    right += sec("Risk Assessment")

    # Risk badge — stacked layout, no nested tables, no clipping
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
        ("ROUNDEDCORNERS",[6]),
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
            ("ROUNDEDCORNERS",[3]),
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

    # Combine columns
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

    # ══════════════════════════════════════════════════════
    # RECOMMENDATIONS
    # ══════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=NAVY4, spaceBefore=4*mm, spaceAfter=4*mm))
    story.append(Paragraph("RECOMMENDATIONS",
                            S("rech", fontName="Helvetica-Bold", fontSize=6.5,
                              textColor=GOLD, leading=9)))
    story.append(Spacer(1, 2*mm))

    recs = []
    if smoker == "yes":
        recs.append(("Quit Smoking",
            "Smoking is the single largest premium driver. Quitting can reduce "
            "annual premiums by 40-60% over time and dramatically improve health."))
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
        ("ROUNDEDCORNERS",[4]),
    ]))
    story.append(rt)

    # ══════════════════════════════════════════════════════
    # FOOTER
    # ══════════════════════════════════════════════════════
    story.append(Spacer(1, 4*mm))
    story.append(HRFlowable(width="100%", thickness=0.4,
                             color=NAVY4, spaceAfter=2*mm))
    ft = Table([[
        Paragraph("PremiumIQ Risk Intelligence  \u00b7  XGBoost Model  \u00b7  R\u00b2 0.8564",
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

    # ── Colour palette ───────────────────────────────────
    NAVY    = colors.HexColor("#050d1a")
    NAVY2   = colors.HexColor("#0a1628")
    NAVY3   = colors.HexColor("#0f1f38")
    GOLD    = colors.HexColor("#c9a84c")
    GOLD_L  = colors.HexColor("#e8c97a")
    PLAT    = colors.HexColor("#d4dae6")
    MUTED   = colors.HexColor("#5a6a82")
    SUCCESS = colors.HexColor("#2fb37a")
    WARNING = colors.HexColor("#e8a838")
    DANGER  = colors.HexColor("#d9534f")
    WHITE   = colors.white

    RISK_COLOR = {"LOW": SUCCESS, "MEDIUM": WARNING, "HIGH": DANGER}[risk_lvl]

    # ── Paragraph styles ─────────────────────────────────
    def S(name, **kw):
        base = dict(fontName="Helvetica", fontSize=9, textColor=PLAT,
                    leading=14, spaceAfter=0, spaceBefore=0)
        base.update(kw)
        return ParagraphStyle(name, **base)

    s_title   = S("title",  fontName="Helvetica-Bold", fontSize=22,
                  textColor=GOLD,    leading=26, alignment=TA_LEFT)
    s_sub     = S("sub",    fontSize=8.5, textColor=MUTED, leading=12)
    s_label   = S("label",  fontName="Helvetica-Bold", fontSize=7,
                  textColor=MUTED,   leading=10, alignment=TA_LEFT)
    s_value   = S("value",  fontName="Helvetica-Bold", fontSize=18,
                  textColor=GOLD_L,  leading=22, alignment=TA_LEFT)
    s_value_s = S("value_s",fontName="Helvetica-Bold", fontSize=13,
                  textColor=GOLD_L,  leading=18, alignment=TA_LEFT)
    s_h2      = S("h2",     fontName="Helvetica-Bold", fontSize=10,
                  textColor=PLAT,    leading=14, spaceBefore=4)
    s_body    = S("body",   fontSize=8.5, textColor=MUTED,  leading=13)
    s_tag     = S("tag",    fontName="Helvetica-Bold", fontSize=7,
                  textColor=WHITE,   leading=9,  alignment=TA_CENTER)
    s_small   = S("small",  fontSize=7.5, textColor=MUTED,  leading=11)
    s_risk    = S("risk",   fontName="Helvetica-Bold", fontSize=14,
                  textColor=RISK_COLOR, leading=18, alignment=TA_CENTER)

    story = []

    # ══════════════════════════════════════════════════════
    # HEADER BAND
    # ══════════════════════════════════════════════════════
    header_data = [[
        Paragraph("PremiumIQ", s_title),
        Paragraph(f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
                  S("hr", fontSize=7.5, textColor=MUTED, alignment=TA_RIGHT))
    ]]
    header_tbl = Table(header_data, colWidths=[120*mm, 54*mm])
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), NAVY2),
        ("TOPPADDING",  (0,0), (-1,-1), 12),
        ("BOTTOMPADDING",(0,0),(-1,-1), 12),
        ("LEFTPADDING", (0,0), (0,-1),  14),
        ("RIGHTPADDING",(-1,0),(-1,-1), 14),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("ROUNDEDCORNERS", [8]),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 4*mm))

    sub_data = [[
        Paragraph("INSURANCE RISK INTELLIGENCE REPORT", S("sr", fontSize=7,
                  textColor=MUTED, fontName="Helvetica-Bold",
                  letterSpacing=2, alignment=TA_LEFT)),
        Paragraph("XGBoost Model  ·  R<super>2</super> 85.64%",
                  S("sr2", fontSize=7.5, textColor=MUTED, alignment=TA_RIGHT))
    ]]
    sub_tbl = Table(sub_data, colWidths=[120*mm, 54*mm])
    sub_tbl.setStyle(TableStyle([
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("TOPPADDING",   (0,0),(-1,-1), 0),
    ]))
    story.append(sub_tbl)
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#172540"), spaceAfter=5*mm))

    # ══════════════════════════════════════════════════════
    # KPI ROW  — 5 cards
    # ══════════════════════════════════════════════════════
    def kpi_cell(label, value, sub):
        return [
            Paragraph(label.upper(), s_label),
            Paragraph(value,         s_value_s),
            Paragraph(sub,           s_small),
        ]

    kpi_data = [[
        kpi_cell("Annual Premium",  f"Rs.{pred:,.0f}",    f"Monthly Rs.{monthly:,.0f}"),
        kpi_cell("BMI Index",       f"{bmi:.1f}",         bmi_cat),
        kpi_cell("Risk Level",      risk_lvl,             f"Score {risk_score}/90"),
        kpi_cell("Monthly Cost",    f"Rs.{monthly:,.0f}", "12 installments"),
        kpi_cell("Smoker",          "Yes" if smoker=="yes" else "No",
                 "+40 pts" if smoker=="yes" else "0 pts"),
    ]]
    kpi_tbl = Table(kpi_data, colWidths=[34.8*mm]*5, rowHeights=None)
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), NAVY3),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("LINEAFTER",     (0,0), (3,-1),  0.5, colors.HexColor("#172540")),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("ROUNDEDCORNERS",[6]),
    ]))
    story.append(kpi_tbl)
    story.append(Spacer(1, 5*mm))

    # ══════════════════════════════════════════════════════
    # TWO COLUMN BODY
    # ══════════════════════════════════════════════════════
    # ── LEFT: Client Profile table ────────────────────────
    def section_head(text):
        return [
            Paragraph(text.upper(),
                      S("sh", fontName="Helvetica-Bold", fontSize=7,
                        textColor=MUTED, letterSpacing=1.5)),
            HRFlowable(width="100%", thickness=0.4,
                       color=colors.HexColor("#172540"), spaceAfter=3*mm)
        ]

    left_story = []
    left_story += section_head("Client Profile")

    profile_rows = [
        ["Age",              str(age)],
        ["Height / Weight",  f"{(bmi * 1.72**2)**0.5:.0f} kg  (BMI {bmi:.1f})"],
        ["BMI Category",     bmi_cat],
        ["Dependents",       str(children)],
        ["Biological Sex",   sex.title()],
        ["Smoking Status",   "Active Smoker" if smoker=="yes" else "Non-Smoker"],
        ["Region",           region.replace("north","North ").replace("south","South ")
                                   .replace("east","East").replace("west","West").title()],
    ]
    prof_tbl = Table(
        [[Paragraph(r, S("pl", fontSize=8, textColor=MUTED)),
          Paragraph(v, S("pv", fontSize=8, fontName="Helvetica-Bold", textColor=PLAT))]
         for r, v in profile_rows],
        colWidths=[30*mm, 44*mm]
    )
    prof_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), NAVY2),
        ("ROWBACKGROUNDS",(0,0), (-1,-1), [NAVY2, NAVY3]),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("ROUNDEDCORNERS",[4]),
    ]))
    left_story.append(prof_tbl)
    left_story.append(Spacer(1, 4*mm))

    # ── LEFT: Premium breakdown ───────────────────────────
    left_story += section_head("Premium Breakdown")
    breakdown_rows = [
        ["Annual Premium",    f"Rs.{pred:,.0f}"],
        ["Monthly Premium",   f"Rs.{monthly:,.0f}"],
        ["Quarterly Premium", f"Rs.{pred/4:,.0f}"],
        ["Semi-Annual",       f"Rs.{pred/2:,.0f}"],
    ]
    bk_tbl = Table(
        [[Paragraph(r, S("bl", fontSize=8, textColor=MUTED)),
          Paragraph(v, S("bv", fontSize=8, fontName="Helvetica-Bold",
                         textColor=GOLD_L, alignment=TA_RIGHT))]
         for r, v in breakdown_rows],
        colWidths=[44*mm, 30*mm]
    )
    bk_tbl.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0,0),(-1,-1), [NAVY2, NAVY3]),
        ("TOPPADDING",     (0,0),(-1,-1), 6),
        ("BOTTOMPADDING",  (0,0),(-1,-1), 6),
        ("LEFTPADDING",    (0,0),(-1,-1), 8),
        ("RIGHTPADDING",   (0,0),(-1,-1), 8),
        ("ROUNDEDCORNERS", [4]),
    ]))
    left_story.append(bk_tbl)

    # ── RIGHT: Risk Assessment ────────────────────────────
    right_story = []
    right_story += section_head("Risk Assessment")

    # Risk level badge
    risk_badge = Table(
        [[Paragraph(f"RISK LEVEL: {risk_lvl}", s_risk),
          Paragraph(f"Score: {risk_score} / 90",
                    S("rs", fontSize=10, textColor=RISK_COLOR,
                      alignment=TA_RIGHT, fontName="Helvetica-Bold"))]],
        colWidths=[35*mm, 35*mm]
    )
    risk_badge.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), NAVY3),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("RIGHTPADDING",  (0,0),(-1,-1), 10),
        ("ROUNDEDCORNERS",[6]),
    ]))
    right_story.append(risk_badge)
    right_story.append(Spacer(1, 4*mm))

    # Risk factors
    if fhi:
        right_story += section_head("High Risk Factors")
        for label, desc in fhi:
            t = Table([[
                Paragraph(label, S("rl", fontSize=8, fontName="Helvetica-Bold",
                                   textColor=DANGER)),
                Paragraph(desc,  S("rd", fontSize=7.5, textColor=MUTED)),
            ]], colWidths=[30*mm, 40*mm])
            t.setStyle(TableStyle([
                ("BACKGROUND",   (0,0),(-1,-1), NAVY3),
                ("TOPPADDING",   (0,0),(-1,-1), 5),
                ("BOTTOMPADDING",(0,0),(-1,-1), 5),
                ("LEFTPADDING",  (0,0),( 0,-1), 6),
                ("LEFTBORDER",   (0,0),( 0,-1), 3, DANGER),
                ("LINEBEFORE",   (0,0),( 0,-1), 3, DANGER),
            ]))
            right_story.append(t)
            right_story.append(Spacer(1, 1.5*mm))

    if fmed:
        right_story += section_head("Moderate Risk Factors")
        for label, desc in fmed:
            t = Table([[
                Paragraph(label, S("ml", fontSize=8, fontName="Helvetica-Bold",
                                   textColor=WARNING)),
                Paragraph(desc,  S("md", fontSize=7.5, textColor=MUTED)),
            ]], colWidths=[30*mm, 40*mm])
            t.setStyle(TableStyle([
                ("BACKGROUND",   (0,0),(-1,-1), NAVY3),
                ("TOPPADDING",   (0,0),(-1,-1), 5),
                ("BOTTOMPADDING",(0,0),(-1,-1), 5),
                ("LEFTPADDING",  (0,0),( 0,-1), 6),
                ("LINEBEFORE",   (0,0),( 0,-1), 3, WARNING),
            ]))
            right_story.append(t)
            right_story.append(Spacer(1, 1.5*mm))

    if flo:
        right_story += section_head("Positive Factors")
        for label, desc in flo:
            t = Table([[
                Paragraph(label, S("ll", fontSize=8, fontName="Helvetica-Bold",
                                   textColor=SUCCESS)),
                Paragraph(desc,  S("ld", fontSize=7.5, textColor=MUTED)),
            ]], colWidths=[30*mm, 40*mm])
            t.setStyle(TableStyle([
                ("BACKGROUND",   (0,0),(-1,-1), NAVY3),
                ("TOPPADDING",   (0,0),(-1,-1), 5),
                ("BOTTOMPADDING",(0,0),(-1,-1), 5),
                ("LEFTPADDING",  (0,0),( 0,-1), 6),
                ("LINEBEFORE",   (0,0),( 0,-1), 3, SUCCESS),
            ]))
            right_story.append(t)
            right_story.append(Spacer(1, 1.5*mm))

    # ── Combine into 2 columns ────────────────────────────
    from reportlab.platypus import KeepInFrame
    left_frame  = KeepInFrame(84*mm, 180*mm, left_story,  mode="shrink")
    right_frame = KeepInFrame(84*mm, 180*mm, right_story, mode="shrink")

    body_tbl = Table([[left_frame, right_frame]],
                     colWidths=[87*mm, 87*mm])
    body_tbl.setStyle(TableStyle([
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
        ("LEFTPADDING",  (0,0),(-1,-1), 0),
        ("RIGHTPADDING", (0,0),(-1,-1), 0),
        ("TOPPADDING",   (0,0),(-1,-1), 0),
        ("BOTTOMPADDING",(0,0),(-1,-1), 0),
        ("LINEAFTER",    (0,0),(0,-1),  0.5, colors.HexColor("#172540")),
        ("RIGHTPADDING", (0,0),(0,-1),  6),
        ("LEFTPADDING",  (1,0),(1,-1),  6),
    ]))
    story.append(body_tbl)
    story.append(Spacer(1, 5*mm))

    # ══════════════════════════════════════════════════════
    # RECOMMENDATIONS
    # ══════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#172540"),
                             spaceBefore=2*mm, spaceAfter=4*mm))

    rec_head = Table([[
        Paragraph("RECOMMENDATIONS", S("rh", fontName="Helvetica-Bold",
                  fontSize=7, textColor=MUTED, letterSpacing=1.5)),
    ]], colWidths=[174*mm])
    story.append(rec_head)
    story.append(Spacer(1, 2*mm))

    recommendations = []
    if smoker == "yes":
        recommendations.append(("Quit Smoking",
            "Smoking is the single largest premium driver. Quitting can reduce "
            "annual premiums by 40-60% over time and dramatically lower health risks."))
    if bmi >= 30:
        recommendations.append(("Reduce BMI",
            f"Your BMI of {bmi:.1f} places you in the {bmi_cat} category. "
            "Achieving a healthy BMI (18.5-24.9) can reduce premiums significantly."))
    if age >= 40:
        recommendations.append(("Early Preventive Care",
            "Regular health check-ups and preventive screenings are advised "
            "for your age bracket to manage long-term insurance costs."))
    if not recommendations:
        recommendations.append(("Maintain Healthy Lifestyle",
            "Your current profile shows low risk factors. Continue healthy habits "
            "to maintain favourable premium rates."))

    rec_rows = []
    for i, (title, body) in enumerate(recommendations):
        rec_rows.append([
            Paragraph(f"{str(i+1).zfill(2)}", S("rn", fontName="Helvetica-Bold",
                      fontSize=11, textColor=GOLD, alignment=TA_CENTER)),
            Paragraph(title, S("rt", fontName="Helvetica-Bold",
                      fontSize=8.5, textColor=PLAT)),
            Paragraph(body,  S("rb", fontSize=8, textColor=MUTED, leading=12)),
        ])

    rec_tbl = Table(rec_rows, colWidths=[12*mm, 40*mm, 122*mm])
    rec_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), NAVY2),
        ("TOPPADDING",    (0,0),(-1,-1), 8),
        ("BOTTOMPADDING", (0,0),(-1,-1), 8),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("RIGHTPADDING",  (0,0),(-1,-1), 8),
        ("VALIGN",        (0,0),(-1,-1), "TOP"),
        ("LINEBEFORE",    (0,0),(0,-1),  3, GOLD),
        ("LINEBELOW",     (0,0),(-1,-2), 0.4, colors.HexColor("#172540")),
        ("ROUNDEDCORNERS",[4]),
    ]))
    story.append(rec_tbl)

    # ══════════════════════════════════════════════════════
    # FOOTER
    # ══════════════════════════════════════════════════════
    story.append(Spacer(1, 4*mm))
    story.append(HRFlowable(width="100%", thickness=0.4,
                             color=colors.HexColor("#172540"), spaceAfter=2*mm))
    footer_data = [[
        Paragraph("PremiumIQ Risk Intelligence  ·  XGBoost Model  ·  R² 0.8564",
                  S("fl", fontSize=7, textColor=MUTED)),
        Paragraph("Estimates are for analytical purposes only and do not constitute financial advice.",
                  S("fr", fontSize=7, textColor=MUTED, alignment=TA_RIGHT)),
    ]]
    footer_tbl = Table(footer_data, colWidths=[87*mm, 87*mm])
    footer_tbl.setStyle(TableStyle([
        ("TOPPADDING",    (0,0),(-1,-1), 0),
        ("BOTTOMPADDING", (0,0),(-1,-1), 0),
    ]))
    story.append(footer_tbl)

    # ── Build ─────────────────────────────────────────────
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

    age    = st.slider("Age", 18, 100, 32)
    height = st.number_input("Height (m)", 1.0, 2.5, 1.72, step=0.01, format="%.2f")
    weight = st.number_input("Weight (kg)", 30.0, 200.0, 72.0, step=0.5)
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

    st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">Demographics</div>', unsafe_allow_html=True)
    children = st.slider("Dependents", 0, 10, 0)
    sex      = st.selectbox("Biological Sex", ["male", "female"])
    smoker   = st.selectbox("Smoking Status", ["no","yes"],
                             format_func=lambda x: "Non-Smoker" if x=="no" else "Active Smoker")
    region   = st.selectbox("Region", ["northeast","northwest","southeast","southwest"],
                             format_func=lambda x: x.replace("north","North ").replace("south","South ")
                                                    .replace("east","East").replace("west","West").title())


    # ── PDF Download ──────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if REPORTLAB_AVAILABLE:
        pdf_bytes = generate_pdf(
            age, height, weight, bmi, bmi_cat, children, sex, smoker, region,
            predict(age, bmi, children, sex, smoker, region),
            predict(age, bmi, children, sex, smoker, region) / 12,
            *risk_profile(age, bmi, smoker)
        )
        st.download_button(
            label="⬇  Download Client Report (PDF)",
            data=pdf_bytes,
            file_name=f"PremiumIQ_Report_{age}y_{region}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
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

# ─────────────────────────────────────────
# COMPUTE
# ─────────────────────────────────────────
pred    = predict(age, bmi, children, sex, smoker, region)
monthly = pred / 12
risk_lvl, risk_score, fhi, fmed, flo = risk_profile(age, bmi, smoker)
risk_color = {"LOW": C["success"], "MEDIUM": C["warning"], "HIGH": C["danger"]}[risk_lvl]


# ─────────────────────────────────────────
# MASTHEAD
# ─────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:flex-end;
            padding:4px 0 28px 0;border-bottom:1px solid rgba(255,255,255,0.07);margin-bottom:32px;">
  <div>
    <div style="font-family:'Playfair Display',serif;font-size:1.9rem;font-weight:700;
                color:#d4dae6;letter-spacing:-0.3px;line-height:1.15;">
      PremiumIQ — Insurance Risk Dashboard
    </div>
    <div style="color:#5a6a82;font-size:0.8rem;margin-top:6px;letter-spacing:0.3px;">
      Predictive analytics powered by XGBoost &nbsp;·&nbsp; Adjust client profile in the sidebar
    </div>
  </div>
  <div style="text-align:right;">
    <div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;">Model Accuracy</div>
    <div style="font-family:'DM Mono',monospace;font-size:1.5rem;color:#c9a84c;font-weight:500;">85.64%</div>
    <div style="font-size:0.68rem;color:#5a6a82;">R² Score</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# KPI STRIP
# ─────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5, gap="small")
with k1: st.markdown(kpi("Annual Premium",    f"₹{pred:,.0f}",       f"Monthly ₹{monthly:,.0f}"), unsafe_allow_html=True)
with k2: st.markdown(kpi("BMI Index",         f"{bmi:.1f}",          bmi_cat, bmi_col), unsafe_allow_html=True)
with k3: st.markdown(kpi("Risk Level",        risk_lvl,              f"Score {risk_score}/90", risk_color), unsafe_allow_html=True)
with k4: st.markdown(kpi("Monthly Cost",      f"₹{monthly:,.0f}",    "÷ 12 installments"), unsafe_allow_html=True)
with k5: st.markdown(kpi("Smoker Surcharge",  "Yes" if smoker=="yes" else "None",
                          "+40 risk pts" if smoker=="yes" else "0 pts added",
                          C["danger"] if smoker=="yes" else C["success"]), unsafe_allow_html=True)

st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
T1, T2, T3, T4, T5 = st.tabs([
    "  Prediction Overview  ",
    "  Model Benchmark  ",
    "  Scenario Simulator  ",
    "  Risk Intelligence  ",
    "  SHAP Explainability  "
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
            ("Age", f"{age} years"), ("Height", f"{height:.2f} m"),
            ("Weight", f"{weight:.1f} kg"), ("BMI", f"{bmi:.2f} — {bmi_cat}"),
            ("Dependents", str(children)), ("Sex", sex.title()),
            ("Smoker", "Yes" if smoker=="yes" else "No"),
            ("Region", region.replace("north","North ").replace("south","South ").replace("east","East").replace("west","West").title())
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

        feats   = ["Smoker", "Age", "BMI", "Children", "Sex", "NW Region", "SE Region", "SW Region"]
        imps    = [0.698, 0.151, 0.063, 0.031, 0.018, 0.013, 0.014, 0.012]
        f_cols  = [C["danger"] if v>0.1 else C["warning"] if v>0.03 else C["muted"] for v in imps]

        fig, ax = plt.subplots(figsize=(5.5, 4.2))
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
        ax.set_title("XGBoost Feature Importance", fontsize=9.5, color=C["plat"], pad=12, fontweight='600')
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
        st.pyplot(fig, use_container_width=True)
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
    model_data = pd.DataFrame({
        "Model":    ["Linear Regression","Decision Tree","Random Forest","AdaBoost","Gradient Boosting","XGBoost"],
        "R²":       [0.8047, 0.7830, 0.8467, 0.7266, 0.7431, 0.8564],
        "RMSE":     [0.4190, 0.4417, 0.3713, 0.4890, 0.4760, 0.3610],
        "Selected": [False, False, False, False, False, True]
    })
    top, bot = st.columns([3, 2], gap="large")

    with top:
        st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">Performance Metrics — All Models</div>', unsafe_allow_html=True)
        fig = plt.figure(figsize=(10, 4.5))
        fig.patch.set_facecolor(C["bg2"])
        gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        fig_style(fig, [ax1, ax2])
        names  = model_data["Model"].tolist()
        r2s    = model_data["R²"].tolist()
        rmses  = model_data["RMSE"].tolist()
        x      = np.arange(len(names))
        bar_c  = [C["gold"] if s else C["blue"] for s in model_data["Selected"]]

        bars1 = ax1.bar(x, r2s, color=bar_c, width=0.6, edgecolor="none", zorder=3)
        ax1.set_xticks(x)
        ax1.set_xticklabels([n.replace(" ","\n") for n in names], fontsize=7, color=C["muted"])
        ax1.set_ylabel("R² Score", fontsize=8)
        ax1.set_ylim(0.68, 0.90)
        ax1.set_title("R² Score  ·  Higher is Better", fontsize=9, color=C["plat"], pad=10, fontweight='600')
        ax1.axhline(max(r2s), color=C["gold"], lw=0.8, ls='--', alpha=0.4)
        for b,v in zip(bars1, r2s):
            ax1.text(b.get_x()+b.get_width()/2, v+0.002, f'{v:.3f}',
                     ha='center', va='bottom', fontsize=7, color=C["plat"])

        bars2 = ax2.bar(x, rmses, color=bar_c, width=0.6, edgecolor="none", zorder=3)
        ax2.set_xticks(x)
        ax2.set_xticklabels([n.replace(" ","\n") for n in names], fontsize=7, color=C["muted"])
        ax2.set_ylabel("RMSE", fontsize=8)
        ax2.set_title("RMSE  ·  Lower is Better", fontsize=9, color=C["plat"], pad=10, fontweight='600')
        ax2.axhline(min(rmses), color=C["gold"], lw=0.8, ls='--', alpha=0.4)
        for b,v in zip(bars2, rmses):
            ax2.text(b.get_x()+b.get_width()/2, v+0.003, f'{v:.3f}',
                     ha='center', va='bottom', fontsize=7, color=C["plat"])

        plt.tight_layout(pad=2)
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
          <div style="font-size:0.72rem;color:#c9a84c;font-weight:600;margin-bottom:8px;">Why XGBoost was selected</div>
          <div style="font-size:0.78rem;color:#5a6a82;line-height:1.65;">
            Achieved the highest R² <span style='color:#d4dae6;font-family:"DM Mono",monospace;'>(0.8564)</span> and
            lowest RMSE <span style='color:#d4dae6;font-family:"DM Mono",monospace;'>(0.3610)</span>. Its gradient
            boosting framework captures non-linear feature interactions — particularly the
            <strong style='color:#d4dae6;'>smoking × BMI</strong> relationship — that linear models cannot express.
          </div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 3 — SCENARIO SIMULATOR
# ════════════════════════════════════════════
with T3:
    st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;">Scenario Parameters</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#5a6a82;font-size:0.8rem;margin-bottom:20px;">Compare a hypothetical profile against your sidebar inputs in real time.</p>', unsafe_allow_html=True)

    sc1,sc2,sc3 = st.columns(3, gap="medium")
    with sc1:
        s_age    = st.slider("Age", 18, 100, age, key="s_age")
        s_smoker = st.selectbox("Smoking", ["no","yes"], index=0 if smoker=="no" else 1,
                                format_func=lambda x: "Non-Smoker" if x=="no" else "Active Smoker", key="s_smk")
    with sc2:
        s_bmi   = st.slider("BMI", 10.0, 55.0, float(round(bmi,1)), step=0.5, key="s_bmi")
        s_child = st.slider("Dependents", 0, 10, children, key="s_ch")
    with sc3:
        s_sex    = st.selectbox("Sex", ["male","female"], index=0 if sex=="male" else 1, key="s_sex")
        s_region = st.selectbox("Region", ["northeast","northwest","southeast","southwest"],
                                index=["northeast","northwest","southeast","southwest"].index(region), key="s_reg")

    sim_pred = predict(s_age, s_bmi, s_child, s_sex, s_smoker, s_region)
    delta    = sim_pred - pred
    delta_p  = (delta / pred) * 100
    dcol     = C["danger"] if delta > 0 else C["success"]
    sign     = "+" if delta > 0 else ""

    st.markdown("<br>", unsafe_allow_html=True)
    d1,d2,d3,d4 = st.columns(4, gap="small")
    with d1: st.markdown(kpi("Baseline Premium",  f"₹{pred:,.0f}",      "Sidebar profile"), unsafe_allow_html=True)
    with d2: st.markdown(kpi("Scenario Premium",  f"₹{sim_pred:,.0f}",  "Simulated profile"), unsafe_allow_html=True)
    with d3: st.markdown(kpi("Absolute Change",   f"{sign}₹{abs(delta):,.0f}", f"{sign}{delta_p:.1f}%", dcol), unsafe_allow_html=True)
    with d4: st.markdown(kpi("Scenario Monthly",  f"₹{sim_pred/12:,.0f}", "÷ 12"), unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">Sensitivity Curves</div>', unsafe_allow_html=True)

    age_r  = range(20, 71, 5)
    age_p  = [predict(a, s_bmi, s_child, s_sex, s_smoker, s_region) for a in age_r]
    bmi_r  = np.arange(17, 46, 1.5)
    bmi_p  = [predict(s_age, b, s_child, s_sex, s_smoker, s_region) for b in bmi_r]
    smk_ns = [predict(a, s_bmi, s_child, s_sex, "no",  s_region) for a in age_r]
    smk_s  = [predict(a, s_bmi, s_child, s_sex, "yes", s_region) for a in age_r]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
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

    axes[2].plot(list(age_r), smk_ns, color=C["success"], lw=2, label="Non-Smoker", zorder=3)
    axes[2].plot(list(age_r), smk_s,  color=C["danger"],  lw=2, label="Smoker",     zorder=3)
    axes[2].fill_between(list(age_r), smk_ns, smk_s, alpha=0.07, color=C["warning"])
    axes[2].axvline(s_age, color=C["muted"], ls='--', lw=0.8, alpha=0.5)
    axes[2].set_xlabel("Age", fontsize=8); axes[2].set_ylabel("₹ Premium", fontsize=8)
    axes[2].set_title("Smoker vs Non-Smoker Gap", fontsize=9, pad=8, fontweight='600')
    leg = axes[2].legend(fontsize=7, framealpha=0)
    for t in leg.get_texts(): t.set_color(C["plat"])

    plt.tight_layout(pad=2.5)
    st.pyplot(fig, use_container_width=True)
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

        fig, ax = plt.subplots(figsize=(5, 3.8))
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
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin:12px 0 12px 0;">Actionable Recommendations</div>', unsafe_allow_html=True)
        recs = []
        if smoker=="yes":  recs.append(("Cessation Program",    "Smoking cessation is the highest-impact action. Estimated premium reduction: 35–45%."))
        if bmi >= 30:      recs.append(("Weight Management",    f"Reducing BMI from {bmi:.1f} to below 25 could lower premiums by 10–18%."))
        elif bmi >= 25:    recs.append(("Maintain Weight",      f"BMI {bmi:.1f} is borderline. Gradual reduction will have a measurable effect."))
        if age >= 40:      recs.append(("Preventive Screenings","Annual checkups at 40+ preempt costly medical events and demonstrate low-risk behaviour."))
        if not recs:       recs.append(("Maintain Lifestyle",   "Excellent profile. No significant changes required to optimise premium costs."))
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
    st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;">AI Explainability — SHAP Analysis</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#5a6a82;font-size:0.8rem;margin-bottom:24px;">SHAP (SHapley Additive exPlanations) reveals exactly <em>why</em> the model predicted a specific premium — showing how much each feature pushed the prediction up or down.</p>', unsafe_allow_html=True)

    if not SHAP_AVAILABLE:
        st.markdown("""
        <div style="padding:24px;background:rgba(232,168,56,0.08);border:1px solid rgba(232,168,56,0.25);
                    border-radius:14px;text-align:center;">
          <div style="font-size:1rem;color:#e8a838;font-weight:600;margin-bottom:8px;">⚠ SHAP library not installed</div>
          <div style="color:#5a6a82;font-size:0.85rem;">Run: <code style="background:rgba(255,255,255,0.06);padding:2px 8px;border-radius:4px;">pip install shap</code> then restart the app.</div>
        </div>""", unsafe_allow_html=True)
    else:
        # ── Build current user input vector ──────────────────────────────
        input_dict = {"age": age, "bmi": bmi, "children": children,
                      "sex": sex, "smoker": smoker, "region": region}
        input_df = pd.get_dummies(pd.DataFrame([input_dict]))
        for c in model_columns:
            if c not in input_df.columns: input_df[c] = 0
        input_df = input_df[model_columns]
        input_df.columns = [str(c) for c in input_df.columns]
        input_df = input_df.astype(np.float64)

        if scaler:
            input_scaled = pd.DataFrame(
                scaler.transform(input_df),
                columns=input_df.columns
            )
        else:
            input_scaled = input_df.copy()

        # ── Load explainer (cached — only runs once) ──────────────────────
        explainer, X_bg = load_shap_explainer(model, model_columns, scaler)

        if explainer is None:
            st.warning("SHAP explainer could not be initialised. Check the error above.")
        else:
            # ── Auto compute SHAP ─────────────────────────────────────────
            with st.spinner("Computing SHAP explainability... (~15 seconds)"):
                shap_obj_all  = explainer(X_bg)
                shap_obj_user = explainer(input_scaled)

            shap_vals_all  = shap_obj_all.values
            shap_user_flat = shap_obj_user.values.flatten()
            base_value     = float(np.array(shap_obj_user.base_values).flatten()[0])

            base_rs  = float(np.exp(base_value))
            final_rs = float(pred)

            # ── Exact ₹ waterfall conversion ─────────────────────────────
            # Only mathematically exact method for log-scale models:
            # add features one-by-one in order of importance, measure
            # the ₹ jump at each step. Last running total == final_rs exactly.
            order      = np.argsort(np.abs(shap_user_flat))[::-1]
            shap_rs    = np.zeros(len(shap_user_flat))
            running    = base_value
            for i in order:
                prev_rs    = np.exp(running)
                running   += shap_user_flat[i]
                shap_rs[i] = np.exp(running) - prev_rs

            # Global: mean absolute ₹ impact per feature across background
            mean_shap_rs = np.zeros(shap_vals_all.shape[1])
            for i in range(shap_vals_all.shape[1]):
                bg_order   = np.argsort(np.abs(shap_vals_all), axis=1)[:, ::-1]
                bg_impacts = np.zeros(shap_vals_all.shape[0])
                for s in range(shap_vals_all.shape[0]):
                    run = base_value
                    for j in bg_order[s]:
                        p   = np.exp(run)
                        run += shap_vals_all[s, j]
                        if j == i:
                            bg_impacts[s] = abs(np.exp(run) - p)
                            break
                mean_shap_rs[i] = bg_impacts.mean()

            # ── Friendly feature labels ───────────────────────────────────
            label_map = {
                "age": "Age", "bmi": "BMI", "children": "Dependents",
                "sex_male": "Sex (Male)", "smoker_yes": "Smoker",
                "region_northwest": "Region: NW", "region_southeast": "Region: SE",
                "region_southwest": "Region: SW"
            }
            feat_labels = [label_map.get(c, c) for c in model_columns]

            sorted_idx = np.argsort(np.abs(shap_rs))[::-1]

            # ── Layout ────────────────────────────────────────────────────
            sh_l, sh_r = st.columns([3, 2], gap="large")

            # ════ LEFT — Rupee waterfall chart ════════════════════════════
            with sh_l:
                st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">Premium Breakdown in ₹ — This Client</div>', unsafe_allow_html=True)

                top_n      = min(8, len(sorted_idx))
                top_idx    = sorted_idx[:top_n][::-1]
                top_rs     = shap_rs[top_idx]
                top_labels = [feat_labels[i] for i in top_idx]

                fig, ax = plt.subplots(figsize=(7, 4.5))
                fig.patch.set_facecolor(C["bg2"])
                ax.set_facecolor(C["bg2"])

                bar_colors = [C["danger"] if v > 0 else C["success"] for v in top_rs]
                bars = ax.barh(top_labels, top_rs, color=bar_colors,
                               height=0.55, edgecolor="none", zorder=3)

                for bar, val in zip(bars, top_rs):
                    sign = "+" if val >= 0 else ""
                    xpos = val + (max(abs(top_rs)) * 0.01 if val >= 0 else -max(abs(top_rs)) * 0.01)
                    ha   = "left" if val >= 0 else "right"
                    ax.text(xpos, bar.get_y() + bar.get_height()/2,
                            f"{sign}₹{abs(val):,.0f}", va="center", ha=ha,
                            color=C["plat"], fontsize=8, fontweight="500")

                ax.axvline(0, color=C["muted"], lw=1, alpha=0.6)
                ax.set_xlabel("Rupee Impact on Premium (₹)", fontsize=8, color=C["muted"])
                ax.set_title("How Each Factor Affects Your Premium (₹)", fontsize=10,
                             color=C["plat"], pad=12, fontweight="600")
                ax.tick_params(colors=C["muted"], labelsize=8.5)
                ax.yaxis.tick_left()
                for sp in ax.spines.values(): sp.set_visible(False)
                ax.grid(axis="x", color=C["border"], lw=0.5, alpha=0.5)
                ax.set_axisbelow(True)

                # Format x-axis as rupees
                ax.xaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}")
                )

                pos_patch = mpatches.Patch(color=C["danger"],  label="Adds to premium")
                neg_patch = mpatches.Patch(color=C["success"], label="Reduces premium")
                leg = ax.legend(handles=[pos_patch, neg_patch], fontsize=7.5,
                                loc="lower right", framealpha=0)
                for t in leg.get_texts(): t.set_color(C["plat"])

                plt.tight_layout(pad=1.5)
                st.pyplot(fig, use_container_width=True)
                plt.close()

                # ── Plain-English explanation — simple format ─────────────
                st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin:20px 0 12px 0;">Plain-English Explanation</div>', unsafe_allow_html=True)

                display_order = np.argsort(np.abs(shap_rs))[::-1][:6]
                step_lines = []
                for i in display_order:
                    lbl   = feat_labels[i]
                    delta = shap_rs[i]
                    if delta >= 0:
                        line = (f'<span style="color:#d9534f;">▲ <strong>{lbl}</strong> '
                                f'added &nbsp;<strong style="font-family:DM Mono,monospace;">'
                                f'+₹{delta:,.0f}</strong> to the premium</span>')
                    else:
                        line = (f'<span style="color:#2fb37a;">▼ <strong>{lbl}</strong> '
                                f'saved &nbsp;<strong style="font-family:DM Mono,monospace;">'
                                f'₹{abs(delta):,.0f}</strong> off the premium</span>')
                    step_lines.append(line)

                st.markdown(f"""
                <div style="padding:18px 20px;background:rgba(255,255,255,0.025);
                            border:1px solid rgba(255,255,255,0.07);border-radius:14px;
                            line-height:2;font-size:0.83rem;color:#8892a4;">
                  Starting from the <strong style="color:#d4dae6;">average customer premium</strong>
                  of <strong style="color:#c9a84c;font-family:'DM Mono',monospace;">₹{base_rs:,.0f}</strong>:
                  <br><br>
                  {"<br>".join(step_lines)}
                  <br>
                  <div style="margin-top:12px;padding-top:12px;
                              border-top:1px solid rgba(255,255,255,0.06);
                              font-size:0.88rem;">
                    ∴ &nbsp;Final estimated premium:&nbsp;
                    <strong style="color:#c9a84c;font-size:1.05rem;
                                   font-family:'DM Mono',monospace;">₹{final_rs:,.0f}</strong>
                  </div>
                </div>""", unsafe_allow_html=True)

            # ════ RIGHT — Global ₹ importance + cards ════════════════════
            with sh_r:
                st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">Average ₹ Impact Per Feature (All Customers)</div>', unsafe_allow_html=True)

                global_idx  = np.argsort(mean_shap_rs)[-8:]
                global_rs   = mean_shap_rs[global_idx]
                global_lbls = [feat_labels[i] for i in global_idx]

                fig2, ax2 = plt.subplots(figsize=(5, 4))
                fig2.patch.set_facecolor(C["bg2"])
                ax2.set_facecolor(C["bg2"])

                bar_cols2 = [C["gold"] if v == global_rs.max() else C["blue"]
                             for v in global_rs]
                ax2.barh(global_lbls, global_rs, color=bar_cols2,
                         height=0.55, edgecolor="none", zorder=3)

                for i, v in enumerate(global_rs):
                    ax2.text(v + max(global_rs)*0.01, i, f"₹{v:,.0f}", va="center",
                             color=C["plat"], fontsize=7.5)

                ax2.set_xlabel("Mean Absolute ₹ Impact", fontsize=8, color=C["muted"])
                ax2.set_title("Avg Premium Impact per Feature", fontsize=9,
                              color=C["plat"], pad=10, fontweight="600")
                ax2.tick_params(colors=C["muted"], labelsize=8)
                ax2.xaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}")
                )
                for sp in ax2.spines.values(): sp.set_visible(False)
                ax2.grid(axis="x", color=C["border"], lw=0.5, alpha=0.5)
                ax2.set_axisbelow(True)

                plt.tight_layout(pad=1.5)
                st.pyplot(fig2, use_container_width=True)
                plt.close()

                # ── Feature contribution cards in ₹ ──────────────────────
                st.markdown('<div style="font-size:0.68rem;color:#5a6a82;text-transform:uppercase;letter-spacing:1.5px;margin:20px 0 12px 0;">This Client — Rupee Contributions</div>', unsafe_allow_html=True)

                for idx in sorted_idx[:6]:
                    lbl       = feat_labels[idx]
                    rs_val    = shap_rs[idx]
                    fval      = input_df.iloc[0][model_columns[idx]]
                    direction = "▲ Adds to premium" if rs_val > 0 else "▼ Reduces premium"
                    d_color   = C["danger"] if rs_val > 0 else C["success"]
                    bar_pct   = min(abs(rs_val) / (np.abs(shap_rs).max() + 1e-9) * 100, 100)
                    sign      = "+" if rs_val > 0 else "-"

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
                        {direction} &nbsp;·&nbsp; Input value: {fval:.2f}
                      </div>
                    </div>""", unsafe_allow_html=True)

                # ── Key insight in ₹ ──────────────────────────────────────
                top_feature = feat_labels[sorted_idx[0]]
                top_rs_val  = shap_rs[sorted_idx[0]]
                top_pct     = abs(top_rs_val) / final_rs * 100

                st.markdown(f"""
                <div style="margin-top:4px;padding:14px 18px;background:rgba(201,168,76,0.06);
                            border:1px solid rgba(201,168,76,0.2);border-radius:12px;">
                  <div style="font-size:0.72rem;color:#c9a84c;font-weight:600;margin-bottom:6px;">
                    🔍 Key Insight
                  </div>
                  <div style="font-size:0.78rem;color:#5a6a82;line-height:1.65;">
                    <strong style="color:#d4dae6;">{top_feature}</strong> is the biggest factor,
                    {"adding" if top_rs_val > 0 else "saving"}
                    <strong style="color:#c9a84c;font-family:'DM Mono',monospace;">
                      ₹{abs(top_rs_val):,.0f}
                    </strong>
                    to this premium — that's
                    <strong style="color:#c9a84c;">{top_pct:.1f}%</strong>
                    of the total predicted amount.
                  </div>
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div style="margin-top:48px;padding:20px 0;border-top:1px solid rgba(255,255,255,0.06);
            display:flex;justify-content:space-between;align-items:center;">
  <div style="font-size:0.7rem;color:#3d4f66;">
    PremiumIQ Risk Intelligence &nbsp;·&nbsp; XGBoost Model &nbsp;·&nbsp; R² 0.8564
  </div>
  <div style="font-size:0.7rem;color:#3d4f66;">
    Estimates are for analytical purposes only and do not constitute financial advice.
  </div>
</div>
""", unsafe_allow_html=True)
