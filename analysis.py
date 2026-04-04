"""
Panama Grid Load Forecasting — Advanced EEE Analytics
======================================================
Author  : [Your Name] | EEE Portfolio Project
Dataset : Panama Case Study — continuous dataset.csv
Model   : XGBoost Regressor with Temporal Split (no data leakage)
Target  : nat_demand (MW)
"""

# ──────────────────────────────────────────────
# 0. IMPORTS
# ──────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# ──────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)
continuous_dataset_path = r"C:\Users\Vikas Vashistha\Downloads\archive (3)\continuous dataset.csv"

df = pd.read_csv(continuous_dataset_path, parse_dates=["datetime"])
df.set_index("datetime", inplace=True)
df.sort_index(inplace=True)

print(f"  Shape          : {df.shape}")
print(f"  Date range     : {df.index.min()} → {df.index.max()}")
print(f"  Target column  : nat_demand")
print(f"  Missing values :\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ──────────────────────────────────────────────
# 2. ADVANCED PREPROCESSING
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Advanced Preprocessing")
print("=" * 60)

# ── 2a. Time-based Interpolation ─────────────────────────────
# Standard ffill/bfill ignores the diurnal curve shape.
# interpolate(method='time') fits values proportionally to the
# time delta — preserving the natural morning ramp and evening peak.
missing_before = df["nat_demand"].isnull().sum()
df["nat_demand"] = df["nat_demand"].interpolate(method="time")
missing_after  = df["nat_demand"].isnull().sum()
print(f"  [Interpolation] Missing nat_demand: {missing_before} → {missing_after}")

# Also interpolate weather features
weather_cols = [c for c in df.columns if c != "nat_demand"]
for col in weather_cols:
    df[col] = df[col].interpolate(method="time")
print(f"  [Interpolation] Weather cols cleaned: {weather_cols}")

# ── 2b. Cyclical Encoding (Hour & Month) ─────────────────────
# A linear encoding (hour=23, hour=0) treats midnight and 11 PM as
# maximally different. Sin/Cos encoding wraps the cycle correctly.
df["hour_sin"]  = np.sin(2 * np.pi * df.index.hour  / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df.index.hour  / 24)
df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
df["dow_sin"]   = np.sin(2 * np.pi * df.index.dayofweek / 7)
df["dow_cos"]   = np.cos(2 * np.pi * df.index.dayofweek / 7)
print("  [Cyclical Encoding] hour, month, day-of-week → sin/cos pairs")

# ── 2c. Physics Feature: Temperature² ────────────────────────
# The demand-temperature curve is U-shaped (high cooling + heating loads
# at extremes). A purely linear T term misses the curvature.
# T² explicitly captures this nonlinearity — a physics-motivated choice.
temp_col = "T2M_toc"   # 2-metre temperature at Tocumen airport (Panama City)
if temp_col in df.columns:
    df["Temperature_Squared"] = df[temp_col] ** 2
    print(f"  [Physics Feature] {temp_col}² created → Temperature_Squared")
else:
    available = [c for c in df.columns if "T2M" in c or "temp" in c.lower()]
    if available:
        temp_col = available[0]
        df["Temperature_Squared"] = df[temp_col] ** 2
        print(f"  [Physics Feature] Used {temp_col}² as proxy")
    else:
        print("  [Physics Feature] WARNING: No temperature column found. Skipping T².")

# ── 2d. Lag Features (Autoregressive Signal) ─────────────────
# Grid operators use previous-day loads as the strongest baseline signal.
df["load_lag_1h"]  = df["nat_demand"].shift(1)   # previous hour
df["load_lag_24h"] = df["nat_demand"].shift(24)  # same hour yesterday
df["load_lag_168h"]= df["nat_demand"].shift(168) # same hour last week
df.dropna(inplace=True)  # remove rows with NaN from lags
print("  [Lag Features] 1h, 24h, 168h lags created")

# ──────────────────────────────────────────────
# 3. FEATURE MATRIX
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Building Feature Matrix")
print("=" * 60)

TARGET = "nat_demand"
EXCLUDE = [TARGET]

FEATURES = [
    # Cyclical time
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    # Physics
    "Temperature_Squared",
    # Autoregressive
    "load_lag_1h", "load_lag_24h", "load_lag_168h",
]

# Add available weather columns dynamically
weather_features = [c for c in df.columns
                    if c not in FEATURES + EXCLUDE
                    and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
FEATURES += weather_features

# Keep only columns that exist
FEATURES = [f for f in FEATURES if f in df.columns]
print(f"  Total features : {len(FEATURES)}")
print(f"  Feature list   : {FEATURES}")

X = df[FEATURES]
y = df[TARGET]

# ──────────────────────────────────────────────
# 4. TEMPORAL TRAIN / TEST SPLIT  (NO SHUFFLE)
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Temporal Train/Test Split (80/20)")
print("=" * 60)

split_idx  = int(len(df) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"  Train : {X_train.index.min()} → {X_train.index.max()}  ({len(X_train):,} samples)")
print(f"  Test  : {X_test.index.min()}  → {X_test.index.max()}   ({len(X_test):,} samples)")
print("  ⚠  No shuffle applied — prevents future data leakage")

# ──────────────────────────────────────────────
# 5. MODEL: XGBoost REGRESSOR
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: XGBoost Model Training")
print("=" * 60)

model = XGBRegressor(
    n_estimators     = 800,
    learning_rate    = 0.05,
    max_depth        = 6,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,    # L1 — sparsity
    reg_lambda       = 1.0,    # L2 — smoothness
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
print("  Training complete.")

# ──────────────────────────────────────────────
# 6. EVALUATION — XGBoost MODEL
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Model Performance")
print("=" * 60)

y_pred = model.predict(X_test)

mae   = mean_absolute_error(y_test, y_pred)
rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
mape  = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
mean_load = y_test.mean()

print(f"\n  ── XGBoost (Our Model) ──────────────────")
print(f"  MAE   : {mae:.2f}  MW")
print(f"  RMSE  : {rmse:.2f} MW")
print(f"  MAPE  : {mape:.2f} %")
print(f"  Mean Load (test set): {mean_load:.2f} MW")

# ──────────────────────────────────────────────
# 7. VALIDATION vs. OFFICIAL PRE-DISPATCH FORECAST
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Validation vs. Official Pre-Dispatch Forecast")
print("=" * 60)

PREDISPATCH_FILE = r"C:\Users\Vikas Vashistha\Downloads\archive (3)\weekly pre-dispatch forecast.csv"

if os.path.exists(PREDISPATCH_FILE):
    pdf = pd.read_csv(PREDISPATCH_FILE)

    # Detect datetime and forecast columns flexibly
    dt_col   = next((c for c in pdf.columns if "date" in c.lower() or "time" in c.lower()), pdf.columns[0])
    fcast_col = next((c for c in pdf.columns if "forecast" in c.lower() or "pred" in c.lower()
                      or "demand" in c.lower()), pdf.columns[1])
    actual_col = next((c for c in pdf.columns
                       if "actual" in c.lower() or "real" in c.lower()
                       or "nat" in c.lower()), None)

    pdf[dt_col] = pd.to_datetime(pdf[dt_col])
    pdf.set_index(dt_col, inplace=True)

    print(f"  Pre-dispatch columns: {list(pdf.columns)}")
    print(f"  Using forecast col  : {fcast_col}")

    # Align with our test set
    overlap = pdf.index.intersection(X_test.index)
    if len(overlap) > 0:
        y_official   = pdf.loc[overlap, fcast_col]
        y_actual_ovl = y_test.loc[overlap]

        off_mae  = mean_absolute_error(y_actual_ovl, y_official)
        off_rmse = np.sqrt(mean_squared_error(y_actual_ovl, y_official))
        off_mape = np.mean(np.abs((y_actual_ovl - y_official) / y_actual_ovl)) * 100

        print(f"\n  ── Official Pre-Dispatch Forecast ───────")
        print(f"  MAE   : {off_mae:.2f}  MW")
        print(f"  RMSE  : {off_rmse:.2f} MW")
        print(f"  MAPE  : {off_mape:.2f} %")

        print(f"\n  ── Comparison Table ─────────────────────")
        print(f"  {'Metric':<10} {'XGBoost':>12} {'Official':>12} {'Better':>10}")
        print(f"  {'-'*46}")
        for metric, xgb_val, off_val in [
            ("MAE (MW)",  mae,  off_mae),
            ("RMSE (MW)", rmse, off_rmse),
            ("MAPE (%)",  mape, off_mape),
        ]:
            winner = "✓ XGBoost" if xgb_val < off_val else "✓ Official"
            print(f"  {metric:<10} {xgb_val:>12.2f} {off_val:>12.2f} {winner:>10}")
    else:
        print("  No temporal overlap between pre-dispatch file and test set dates.")
        print("  Computing pre-dispatch metrics independently...")
        if actual_col:
            y_off_actual   = pdf[actual_col].dropna()
            y_off_forecast = pdf.loc[y_off_actual.index, fcast_col]
            off_mae  = mean_absolute_error(y_off_actual, y_off_forecast)
            off_rmse = np.sqrt(mean_squared_error(y_off_actual, y_off_forecast))
            off_mape = np.mean(np.abs((y_off_actual - y_off_forecast) / y_off_actual)) * 100
            print(f"\n  Official Pre-Dispatch → MAE: {off_mae:.2f} MW | RMSE: {off_rmse:.2f} MW | MAPE: {off_mape:.2f}%")
else:
    print(f"  '{PREDISPATCH_FILE}' not found — skipping official comparison.")
    print("  Place the file in /data and re-run.")

# ──────────────────────────────────────────────
# 8. FEATURE IMPORTANCE
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Feature Importance")
print("=" * 60)

importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(importances.head(10).to_string())

# ──────────────────────────────────────────────
# 9. VISUALISATIONS
# ──────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")
palette = {"blue": "#1F77B4", "orange": "#FF7F0E", "red": "#D62728", "green": "#2CA02C"}

# ── Fig 1: Actual vs Predicted (last 2 weeks of test set) ────
fig, axes = plt.subplots(3, 1, figsize=(16, 14))

sample = X_test.iloc[-336:]   # last 2 weeks (hourly)
y_actual_sample = y_test.iloc[-336:]
y_pred_sample   = model.predict(sample)

axes[0].plot(sample.index, y_actual_sample.values, color=palette["blue"],
             linewidth=1.2, label="Actual Load (nat_demand)")
axes[0].plot(sample.index, y_pred_sample,           color=palette["orange"],
             linewidth=1.2, linestyle="--", label="XGBoost Forecast")
axes[0].set_title("Panama National Grid — Actual vs. XGBoost Forecast (Last 2 Weeks of Test Set)",
                   fontsize=13, fontweight="bold")
axes[0].set_ylabel("Load (MW)")
axes[0].legend()
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))

# ── Fig 2: Residuals ─────────────────────────────────────────
residuals = y_actual_sample.values - y_pred_sample
axes[1].fill_between(sample.index, residuals, 0, where=(residuals >= 0),
                     color=palette["green"], alpha=0.5, label="Under-forecast")
axes[1].fill_between(sample.index, residuals, 0, where=(residuals < 0),
                     color=palette["red"],   alpha=0.5, label="Over-forecast")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("Forecast Residuals (Actual − Predicted)")
axes[1].set_ylabel("Error (MW)")
axes[1].legend()
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

# ── Fig 3: Feature Importance ────────────────────────────────
top_n = importances.head(12)
axes[2].barh(top_n.index[::-1], top_n.values[::-1], color=palette["blue"], edgecolor="white")
axes[2].set_title("XGBoost Feature Importance (Gain)")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
fig.savefig("outputs/forecast_analysis.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n  [Fig 1] Saved: outputs/forecast_analysis.png")

# ── Fig 2: Diurnal Curve by Day Type ────────────────────────
fig2, ax = plt.subplots(figsize=(12, 5))
df["is_weekend"] = df.index.dayofweek >= 5
df["hour"] = df.index.hour
for label, grp in df.groupby("is_weekend"):
    tag = "Weekend" if label else "Weekday"
    color = palette["orange"] if label else palette["blue"]
    hourly = grp.groupby("hour")["nat_demand"].mean()
    ax.plot(hourly.index, hourly.values, label=tag, color=color, linewidth=2)

ax.set_title("Average Diurnal Load Curve — Panama Grid (Weekday vs Weekend)", fontsize=12, fontweight="bold")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Average Load (MW)")
ax.set_xticks(range(0, 24, 2))
ax.legend()
fig2.savefig("outputs/diurnal_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("  [Fig 2] Saved: outputs/diurnal_curve.png")

# ── Fig 3: T² vs Demand scatter ─────────────────────────────
if "Temperature_Squared" in df.columns:
    fig3, ax = plt.subplots(figsize=(8, 5))
    sample_scatter = df.sample(min(5000, len(df)), random_state=42)
    sc = ax.scatter(
        np.sqrt(sample_scatter["Temperature_Squared"]),   # back to °C for x-axis
        sample_scatter["nat_demand"],
        c=sample_scatter.index.hour, cmap="plasma", alpha=0.4, s=8
    )
    plt.colorbar(sc, ax=ax, label="Hour of Day")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("National Demand (MW)")
    ax.set_title("U-Curve: Temperature vs Load (coloured by hour)", fontsize=11, fontweight="bold")
    fig3.savefig("outputs/temperature_ucurve.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("  [Fig 3] Saved: outputs/temperature_ucurve.png")

print("\n" + "=" * 60)
print("ALL STEPS COMPLETE")
print(f"  MAE  = {mae:.2f} MW  |  RMSE = {rmse:.2f} MW  |  MAPE = {mape:.2f}%")
print("=" * 60)
