"""
Panama Grid — Next-Hour Load Forecasting
=========================================
Author  : [Your Name] | EEE Portfolio Project
Dataset : Panama Case Study — continuous_dataset.csv
Model   : XGBoost Regressor (chronological 80/20 split, no data leakage)
Task    : Predict nat_demand (MW) for the NEXT hour
Target  : nat_demand (MW)
"""

# ----------------------------------------------
# 0. IMPORTS
# ----------------------------------------------
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# ----------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------
print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)

# Accept file path from CLI; fall back to conventional relative path.
continuous_dataset_path = sys.argv[1] if len(sys.argv) > 1 else "data/continuous_dataset.csv"

df = pd.read_csv(continuous_dataset_path, parse_dates=["datetime"])
df.set_index("datetime", inplace=True)
df.sort_index(inplace=True)

print(f"  Shape          : {df.shape}")
print(f"  Date range     : {df.index.min()} -> {df.index.max()}")
print(f"  Target column  : nat_demand")
print(f"  Missing values :\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ----------------------------------------------
# 2. PREPROCESSING
# ----------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Preprocessing")
print("=" * 60)

# -- 2a. Target: drop rows with missing nat_demand -------------
# Interpolating the target (nat_demand) would leak information:
# any synthetic value is derived from future observations.
# We simply discard missing target rows to keep the dataset clean.
missing_before = df["nat_demand"].isnull().sum()
df = df.dropna(subset=["nat_demand"])
missing_after  = df["nat_demand"].isnull().sum()
print(f"  [Target] Dropped {missing_before} row(s) with missing nat_demand "
      f"(remaining missing: {missing_after})")

# -- 2b. Weather features: linear interpolation ----------------
# On uniform hourly data, time-based interpolation reduces to
# linear interpolation between the two bracketing observations.
# This is acceptable for short gaps in exogenous weather series.
weather_cols = [c for c in df.columns if c != "nat_demand"]
for col in weather_cols:
    df[col] = df[col].interpolate(method="linear")
print(f"  [Interpolation] Weather cols linearly interpolated: {weather_cols}")

# -- 2c. Cyclical Encoding (Hour, Day-of-Week) -----------------
# Sin/Cos encoding prevents the model treating hour 23 and hour 0
# as maximally different (they are actually adjacent).
df["hour_sin"] = np.sin(2 * np.pi * df.index.hour      / 24)
df["hour_cos"] = np.cos(2 * np.pi * df.index.hour      / 24)
df["dow_sin"]  = np.sin(2 * np.pi * df.index.dayofweek / 7)
df["dow_cos"]  = np.cos(2 * np.pi * df.index.dayofweek / 7)
print("  [Cyclical Encoding] hour, day-of-week -> sin/cos pairs")

# -- 2d. Seasonal Encoding (Month) — only if multi-year dataset -
date_span_years = (df.index.max() - df.index.min()).days / 365.25
USE_MONTH_FEATURES = date_span_years > 1.0
if USE_MONTH_FEATURES:
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    print(f"  [Cyclical Encoding] Dataset spans {date_span_years:.1f} years — "
          f"month sin/cos added")
else:
    print(f"  [Cyclical Encoding] Dataset spans {date_span_years:.1f} years — "
          f"month features omitted (< 1 year)")

# -- 2e. Physics Feature: (T - 24)² ---------------------------
# Panama's climate is warm (roughly 20–35 °C).  T² is minimised at
# 0 °C which lies far outside the observed range and therefore does
# not capture a meaningful U-curve for this dataset.
# (T - 24)² is centred near the thermal-comfort baseline for the
# tropics, so it rises symmetrically as temperatures deviate from
# that reference — a physically motivated quadratic feature.
temp_col = "T2M_toc"   # 2-metre temperature at Tocumen airport (Panama City)
if temp_col in df.columns:
    df["T_dev_sq"] = (df[temp_col] - 24) ** 2
    print(f"  [Physics Feature] (T2M_toc - 24)² created -> T_dev_sq")
else:
    print("  [Physics Feature] WARNING: T2M_toc not found — T_dev_sq skipped.")

# -- 2f. Lag Features (Autoregressive Signal) ------------------
df["load_lag_1h"]   = df["nat_demand"].shift(1)    # previous hour
df["load_lag_24h"]  = df["nat_demand"].shift(24)   # same hour yesterday
df["load_lag_168h"] = df["nat_demand"].shift(168)  # same hour last week
df.dropna(inplace=True)   # remove rows with NaN introduced by lags
print("  [Lag Features] 1h, 24h, 168h lags created")

# ----------------------------------------------
# 3. FEATURE MATRIX
# ----------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Building Feature Matrix")
print("=" * 60)

TARGET = "nat_demand"

# Explicit feature list — no automatic column detection.
# Only well-understood, causally defensible features are included.
FEATURES = [
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "load_lag_1h", "load_lag_24h", "load_lag_168h",
    "T2M_toc",
]

# Add physics quadratic term if it was created
if "T_dev_sq" in df.columns:
    FEATURES.append("T_dev_sq")

# Add seasonal features only when the dataset warrants them
if USE_MONTH_FEATURES:
    FEATURES += ["month_sin", "month_cos"]

# Guard: retain only columns that actually exist
FEATURES = [f for f in FEATURES if f in df.columns]

print(f"  Total features : {len(FEATURES)}")
print(f"  Feature list   : {FEATURES}")

X = df[FEATURES]
y = df[TARGET]

# ----------------------------------------------
# 4. TEMPORAL TRAIN / TEST SPLIT  (NO SHUFFLE)
# ----------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Temporal Train/Test Split (80/20, chronological)")
print("=" * 60)

split_idx       = int(len(df) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"  Train : {X_train.index.min()} -> {X_train.index.max()}  ({len(X_train):,} samples)")
print(f"  Test  : {X_test.index.min()}  -> {X_test.index.max()}   ({len(X_test):,} samples)")
print("  !  No shuffle applied — prevents future-data leakage")

# ----------------------------------------------
# 5. MODEL: XGBoost REGRESSOR
# ----------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: XGBoost Model Training")
print("=" * 60)

model = XGBRegressor(
    n_estimators     = 300,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = 42,
    n_jobs           = -1,
)

model.fit(X_train, y_train, verbose=False)
print("  Training complete.")

# ----------------------------------------------
# 6. EVALUATION — XGBoost MODEL + NAIVE BASELINE
# ----------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: Model Performance")
print("=" * 60)

y_pred    = model.predict(X_test)

mae       = mean_absolute_error(y_test, y_pred)
rmse      = np.sqrt(mean_squared_error(y_test, y_pred))
mape      = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2        = r2_score(y_test, y_pred)
mean_load = y_test.mean()

print(f"\n  -- XGBoost (Our Model) ------------------")
print(f"  MAE   : {mae:.2f}  MW")
print(f"  RMSE  : {rmse:.2f} MW")
print(f"  MAPE  : {mape:.2f} %")
print(f"  R²    : {r2:.4f}")
print(f"  Mean Load (test set): {mean_load:.2f} MW")

# -- Naive baseline: same-hour-yesterday (load_lag_24h) --------
# This is the industry-standard persistence baseline for next-hour
# grid forecasting.  Any useful model should beat it.
y_baseline     = X_test["load_lag_24h"]
baseline_mape  = np.mean(np.abs((y_test - y_baseline) / y_test)) * 100
improvement_pp = baseline_mape - mape   # percentage-point reduction

print(f"\n  -- Naive Baseline (same-hour-yesterday) -")
print(f"  MAPE  : {baseline_mape:.2f} %")
print(f"\n  -- XGBoost vs Baseline ------------------")
print(f"  Improvement : {improvement_pp:+.2f} pp  "
      f"(XGBoost MAPE {mape:.2f}% vs Baseline {baseline_mape:.2f}%)")

# ----------------------------------------------
# 7. VALIDATION vs. OFFICIAL PRE-DISPATCH FORECAST
# ----------------------------------------------
print("\n" + "=" * 60)
print("STEP 7: Validation vs. Official Pre-Dispatch Forecast")
print("=" * 60)

# Exact column names as found in the CSV header (inspected directly):
#   datetime      — timestamp column
#   load_forecast — official pre-dispatch forecast (MW)
PREDISPATCH_FILE      = "data/weekly pre-dispatch forecast.csv"
PREDISPATCH_DT_COL    = "datetime"
PREDISPATCH_FCAST_COL = "load_forecast"

if os.path.exists(PREDISPATCH_FILE):
    pdf = pd.read_csv(PREDISPATCH_FILE, parse_dates=[PREDISPATCH_DT_COL])
    pdf.set_index(PREDISPATCH_DT_COL, inplace=True)
    pdf.sort_index(inplace=True)

    print(f"  Pre-dispatch columns : {list(pdf.columns)}")
    print(f"  Forecast column used : {PREDISPATCH_FCAST_COL}")

    # Restrict comparison to the intersection of the test-set timestamps
    # and the timestamps present in the official forecast file.
    overlap = pdf.index.intersection(X_test.index)
    if len(overlap) > 0:
        y_official   = pdf.loc[overlap, PREDISPATCH_FCAST_COL]
        y_actual_ovl = y_test.loc[overlap]

        off_mae  = mean_absolute_error(y_actual_ovl, y_official)
        off_rmse = np.sqrt(mean_squared_error(y_actual_ovl, y_official))
        off_mape = np.mean(np.abs((y_actual_ovl - y_official) / y_actual_ovl)) * 100

        print(f"\n  -- Official Pre-Dispatch Forecast -------")
        print(f"  Overlap timestamps   : {len(overlap):,}")
        print(f"  MAE   : {off_mae:.2f}  MW")
        print(f"  RMSE  : {off_rmse:.2f} MW")
        print(f"  MAPE  : {off_mape:.2f} %")

        print(f"\n  -- Comparison Table ---------------------")
        print(f"  {'Metric':<10} {'XGBoost':>12} {'Official':>12} {'Better':>10}")
        print(f"  {'-'*46}")
        for metric, xgb_val, off_val in [
            ("MAE (MW)",  mae,  off_mae),
            ("RMSE (MW)", rmse, off_rmse),
            ("MAPE (%)",  mape, off_mape),
        ]:
            winner = "* XGBoost" if xgb_val < off_val else "* Official"
            print(f"  {metric:<10} {xgb_val:>12.2f} {off_val:>12.2f} {winner:>10}")
    else:
        print("  No temporal overlap between pre-dispatch file and test set dates.")
        print("  Skipping comparison — ensure both files cover the same period.")
else:
    print(f"  '{PREDISPATCH_FILE}' not found — skipping official comparison.")
    print("  Place the file at data/weekly pre-dispatch forecast.csv and re-run.")

# ----------------------------------------------
# 8. FEATURE IMPORTANCE
# ----------------------------------------------
print("\n" + "=" * 60)
print("STEP 8: Feature Importance")
print("=" * 60)

importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(importances.head(10).to_string())

# ----------------------------------------------
# 9. VISUALISATIONS
# ----------------------------------------------
os.makedirs("outputs", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")
palette = {"blue": "#1F77B4", "orange": "#FF7F0E", "red": "#D62728", "green": "#2CA02C"}

# -- Fig 1: Actual vs Predicted (last 2 weeks of test set) ----
fig, axes = plt.subplots(3, 1, figsize=(16, 14))

sample          = X_test.iloc[-336:]   # last 2 weeks (hourly)
y_actual_sample = y_test.iloc[-336:]
y_pred_sample   = model.predict(sample)

axes[0].plot(sample.index, y_actual_sample.values, color=palette["blue"],
             linewidth=1.2, label="Actual Load (nat_demand)")
axes[0].plot(sample.index, y_pred_sample,          color=palette["orange"],
             linewidth=1.2, linestyle="--", label="XGBoost Forecast")
axes[0].set_title("Panama National Grid — Actual vs. XGBoost Forecast (Last 2 Weeks of Test Set)",
                   fontsize=13, fontweight="bold")
axes[0].set_ylabel("Load (MW)")
axes[0].legend()
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))

# -- Fig 2: Residuals -----------------------------------------
residuals = y_actual_sample.values - y_pred_sample
axes[1].fill_between(sample.index, residuals, 0, where=(residuals >= 0),
                     color=palette["green"], alpha=0.5, label="Under-forecast")
axes[1].fill_between(sample.index, residuals, 0, where=(residuals < 0),
                     color=palette["red"],   alpha=0.5, label="Over-forecast")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("Forecast Residuals (Actual - Predicted)")
axes[1].set_ylabel("Error (MW)")
axes[1].legend()
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

# -- Fig 3: Feature Importance --------------------------------
top_n = importances.head(12)
axes[2].barh(top_n.index[::-1], top_n.values[::-1], color=palette["blue"], edgecolor="white")
axes[2].set_title("XGBoost Feature Importance (Gain)")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
fig.savefig("outputs/forecast_analysis.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n  [Fig 1] Saved: outputs/forecast_analysis.png")

# -- Fig 2: Diurnal Curve by Day Type ------------------------
fig2, ax = plt.subplots(figsize=(12, 5))
df["is_weekend"] = df.index.dayofweek >= 5
df["hour"]       = df.index.hour
for label, grp in df.groupby("is_weekend"):
    tag   = "Weekend" if label else "Weekday"
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

# -- Fig 3: Temperature vs Demand scatter ---------------------
# Title reads "Temperature vs Demand" — not "U-Curve" — because
# the shape in Panama's warm-climate range is not visually U-shaped.
if temp_col in df.columns:
    fig3, ax = plt.subplots(figsize=(8, 5))
    sample_scatter = df.sample(min(5000, len(df)), random_state=42)
    sc = ax.scatter(
        sample_scatter[temp_col],
        sample_scatter["nat_demand"],
        c=sample_scatter.index.hour, cmap="plasma", alpha=0.4, s=8
    )
    plt.colorbar(sc, ax=ax, label="Hour of Day")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("National Demand (MW)")
    ax.set_title("Temperature vs Demand (coloured by hour)", fontsize=11, fontweight="bold")
    fig3.savefig("outputs/temperature_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("  [Fig 3] Saved: outputs/temperature_scatter.png")

print("\n" + "=" * 60)
print("ALL STEPS COMPLETE")
print(f"  MAE  = {mae:.2f} MW  |  RMSE = {rmse:.2f} MW  |  MAPE = {mape:.2f}%  |  R² = {r2:.4f}")
print(f"  Naive Baseline MAPE = {baseline_mape:.2f}%  |  Improvement = {improvement_pp:+.2f} pp")
print("=" * 60)
