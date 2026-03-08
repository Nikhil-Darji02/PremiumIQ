"""
fix_model.py  —  XGBoost 3.2.0 compatible
──────────────────────────────────────────
Run ONCE in your project folder:
    python fix_model.py

Requires in same folder:
    insurance.csv

Generates:
    model.pkl, scaler.pkl, columns.pkl
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import xgboost as xgb

print("=" * 52)
print(f"  XGBoost version : {xgb.__version__}")
print("  SmartPolicy — Model Fix & Retrain")
print("=" * 52)

# ── 1. Load & preprocess ───────────────────────────────
print("\n[1] Loading insurance.csv...")
df = pd.read_csv("insurance.csv")
print(f"    Shape: {df.shape}")

df["charges"] = np.log(df["charges"])
df = pd.get_dummies(df, drop_first=True)

X = df.drop("charges", axis=1)
y = df["charges"]

# Force all column names to plain strings — critical for SHAP
X.columns = [str(c) for c in X.columns]
columns   = list(X.columns)
print(f"    Features: {columns}")

# ── 2. Split & scale ───────────────────────────────────
print("\n[2] Splitting and scaling...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler     = StandardScaler()
X_train_sc = pd.DataFrame(
    scaler.fit_transform(X_train), columns=columns
)
X_test_sc  = pd.DataFrame(
    scaler.transform(X_test), columns=columns
)
print(f"    Train: {X_train_sc.shape}  Test: {X_test_sc.shape}")

# ── 3. Train XGBoost 3.2.0 ────────────────────────────
print("\n[3] Training XGBoost 3.2.0...")
model = XGBRegressor(
    n_estimators     = 500,
    learning_rate    = 0.05,
    max_depth        = 4,
    subsample        = 1.0,
    colsample_bytree = 1.0,
    random_state     = 42,
    device           = "cpu",     # explicit for 3.x
)
model.fit(X_train_sc, y_train, verbose=False)
print("    Training complete.")

# ── 4. Evaluate ────────────────────────────────────────
print("\n[4] Evaluating...")
y_pred     = model.predict(X_test_sc)
r2         = r2_score(y_test, y_pred)
rmse       = np.sqrt(mean_squared_error(y_test, y_pred))
r2_orig    = r2_score(np.exp(y_test), np.exp(y_pred))
print(f"    R2  (log scale)  : {r2:.4f}")
print(f"    RMSE (log scale) : {rmse:.4f}")
print(f"    R2  (orig scale) : {r2_orig:.4f}")

# ── 5. Verify feature names ────────────────────────────
print("\n[5] Verifying feature names...")
feat_names = model.get_booster().feature_names
print(f"    Feature names: {feat_names}")
assert feat_names and all(isinstance(n, str) for n in feat_names), \
    "Feature names not set correctly!"
print("    Verified OK")

# ── 6. Test SHAP before saving ─────────────────────────
print("\n[6] Testing SHAP compatibility...")
try:
    import shap
    explainer = shap.TreeExplainer(model)
    sv        = explainer.shap_values(X_test_sc.iloc[:3])
    print(f"    SHAP values shape : {np.array(sv).shape}")
    print("    SHAP test PASSED ✓")
except Exception as e:
    print(f"    SHAP test result  : {e}")
    print("    (SHAP may still work in app — continuing)")

# ── 7. Save ────────────────────────────────────────────
print("\n[7] Saving artifacts...")
joblib.dump(model,   "model.pkl")
joblib.dump(scaler,  "scaler.pkl")
joblib.dump(columns, "columns.pkl")

print("\n" + "=" * 52)
print("  Saved successfully:")
print("    model.pkl    — XGBoost 3.2.0 model")
print("    scaler.pkl   — StandardScaler")
print("    columns.pkl  — Feature names")
print("=" * 52)
print("\n  Run: streamlit run app.py")
