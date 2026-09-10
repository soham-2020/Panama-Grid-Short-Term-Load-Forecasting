# Panama Grid Load Forecasting — Interview Preparation & Technical Master Guide

> **Project Summary**: Next-Hour Electricity Demand Forecasting for the National Grid of Panama using Physics-Informed Feature Engineering, XGBoost, and Leakage-Free Temporal Validation.
>
> **Core Result**: **1.63% MAPE**, **$R^2 = 0.9799$**, beating the same-hour-yesterday persistence baseline (6.03% MAPE) by **+4.39 percentage points**.

---

## 1. The 30-Second Elevator Pitch

> *"I built a next-hour electricity demand forecasting engine for the Panama National Grid using 5.5 years of hourly grid and meteorological data. Electricity grids must continuously balance generation with demand to prevent frequency collapse and blackouts. By eliminating target data leakage, engineering tropical climate-specific temperature features and autoregressive lag signals, my XGBoost model achieved a **1.63% MAPE** and an **$R^2$ of 0.98** on an unseen chronological test set, outperforming the industry persistence baseline by **4.39 percentage points**."*

---

## 2. The 2-Minute Deep Dive (Technical Walkthrough)

When an interviewer asks: *"Walk me through the architecture and design decisions of your load forecasting project"*, structure your answer in 5 steps:

1. **Problem Formulation**:
   - Next-hour ($t+1$) national grid load (`nat_demand` in MW).
   - Not 72-hour: Short-Term Load Forecasting (STLF) at the 1-hour horizon directly informs real-time economic dispatch, spinning reserve management, and automatic generation control (AGC).

2. **Data Integrity & Leakage Prevention**:
   - Initial codebase had target leakage (interpolating missing `nat_demand` using future values). I replaced this with strict target dropping (`dropna`).
   - Time-series split: Chronological 80/20 split without shuffling. Shuffling time-series creates catastrophic lookahead bias.

3. **Domain-Specific Feature Engineering**:
   - **Autoregressive terms**: $t-1$h (momentum), $t-24$h (diurnal rhythm), $t-168$h (weekly rhythm).
   - **Cyclical encoding**: Sine/cosine pairs for hour, day of week, and month to remove artificial 23 $\to$ 0 step discontinuities.
   - **Physics term**: Refactored the flawed $T^2$ feature into $(T - 24)^2$, representing quadratic demand deviation from the tropical 24 °C human thermal comfort baseline.

4. **Model Selection & Regularization**:
   - Gradient boosted trees (XGBoost Regressor) with conservative depth (`max_depth=4`, `n_estimators=300`, `learning_rate=0.05`).
   - Resisted deep learning (LSTM/Transformers) because tabular time series with strong autoregressive features are proven to be modeled more accurately, with lower latency and zero hallucination risk, by tree ensembles.

5. **Validation & Benchmarking**:
   - Benchmarked against a naive persistence baseline (same-hour-yesterday: 6.03% MAPE).
   - Verified official pre-dispatch forecast headers directly from the CSV schema (`datetime`, `load_forecast`) and evaluated only on overlapping timestamps.

---

## 3. Key Design Decisions & "Why" Matrix

| Question | Naive / Flawed Approach | Our Production Approach | Engineering & Mathematical Rationale |
|---|---|---|---|
| **Handling Missing Target** | Interpolating `nat_demand` | `df.dropna(subset=['nat_demand'])` | Interpolating target values synthesizes $y_t$ using future labels $y_{t+k}$, creating target data leakage. For exogenous weather features, interpolation is fine, but never for the ground truth target. |
| **Interpolation Claim** | "Time interpolation preserves diurnal curves" | Linear interpolation | On uniformly spaced hourly data ($\Delta t = 1\text{ h}$), time-based interpolation is mathematically identical to linear interpolation. Claiming it "preserves diurnal curves" is inaccurate. |
| **Temperature Feature** | $T^2$ | $(T_{2M} - 24)^2$ | Panama's temperatures range from 20 °C to 35 °C. The function $T^2$ has its vertex at $0\text{ }^\circ\text{C}$. In the 20–35 °C range, $T^2$ is monotonically increasing with no minimum; it is NOT a U-curve. $(T - 24)^2$ is physically centered at the 24 °C thermal comfort setpoint, rising symmetrically as air conditioning loads increase with higher temperatures or decrease with cooler rains. |
| **Feature Selection** | Auto-adding all numeric columns | Explicit 11-feature list | Dynamic column loops risk pulling in identifiers, unverified sensor columns, or future-dated records. Explicit whitelist ensures auditability and causal validity. |
| **Train/Test Split** | Random train/test split or K-Fold | Chronological 80/20 split | Random shuffling allows the model to train on hour $t+1$ and predict hour $t$. In production, time moves forward only. |
| **Baseline Benchmark** | No baseline or comparing to mean | Naive persistence ($t-24$h) | In grid operations, yesterday's load at the same hour is the universal standard. Any model must prove positive lift over this baseline. |
| **Model Choice** | LSTM / Transformer / Deep Learning | XGBoost Regressor | Tabular autoregressive time series do not benefit from sequence-to-sequence deep networks. XGBoost trains in seconds, has low inference latency (<5 ms), handles non-linearities, and provides native feature importance. |

---

## 4. The Exact Feature Set (11 Features)

Every feature in the final model has an electrical or meteorological justification:

```
Total Features = 11:
├── Autoregressive (Lags)
│   ├── load_lag_1h    (42.2% gain) -> Momentum & immediate inertia
│   ├── load_lag_168h   (20.4% gain) -> Weekly periodicity (weekend vs weekday)
│   └── load_lag_24h   (10.3% gain) -> Daily diurnal periodicity
├── Meteorological (Weather)
│   ├── T2M_toc        (13.5% gain) -> 2m Air Temp at Tocumen Airport (°C)
│   └── T_dev_sq        (0.28% gain) -> (T2M_toc - 24)², thermal deviation
└── Cyclical Time Encoding
    ├── hour_sin / hour_cos (3.2% / 4.3%) -> 24-hour circular cycle
    ├── dow_sin / dow_cos   (4.9% / 0.6%) -> 7-day day-of-week cycle
    └── month_sin / month_cos (0.15%)     -> Annual seasonal cycle (spans 5.5 yrs)
```

### Why Sin/Cos Encoding?
If hour is encoded as an integer $0, 1, \dots, 23$:
- The numerical difference between 23:00 (11 PM) and 00:00 (midnight) is $23 - 0 = 23$.
- In reality, these two hours are separated by only 1 hour.
- Sine and Cosine projection:
  $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot h}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot h}{24}\right)$$
- The Euclidean distance between 23:00 and 00:00 becomes:
  $$\sqrt{(\sin(46\pi/24) - 0)^2 + (\cos(46\pi/24) - 1)^2} \approx 0.26$$
  preserving true temporal geometry.

---

## 5. Verified Performance Metrics

Evaluated on the chronological unseen test set (**May 25, 2019 $\to$ June 27, 2020 — 9,576 hourly timesteps**):

| Metric | XGBoost Model | Naive Baseline (Same-Hour-Yesterday) | Lift / Advantage |
|---|---|---|---|
| **MAPE** (Mean Absolute % Error) | **1.63%** | 6.03% | **+4.39 percentage points** |
| **MAE** (Mean Absolute Error) | **19.57 MW** | ~73.1 MW | **~3.7x lower error** |
| **RMSE** (Root Mean Squared Error) | **26.40 MW** | ~98.4 MW | Penalizes peak spikes |
| **$R^2$ Score** | **0.9799** | — | **98.0% variance explained** |
| **Mean Test Load** | 1,211.54 MW | — | Typical range: 900–1,650 MW |

---

## 6. Top 15 Tough Interview Questions & Bulletproof Answers

### Q1: "Why did you build a next-hour model instead of 24-hour or 72-hour ahead?"
**Answer**:
> *"Different grid operations require different forecasting horizons. A 72-hour forecast is used for long-term thermal unit commitment (slow start-up times). However, real-time grid balance, automatic generation control (AGC), and spinning reserve adjustments happen on the 1-hour and 15-minute horizon. Next-hour load forecasting determines immediate ramping requirements and avoids costly imbalance penalties. Furthermore, claiming a 72-hour forecast while feeding 1-hour lag features (`load_lag_1h`) is a classic formulation flaw, because in a true 72-hour forecast, $t-1$h is unknown beyond the first step."*

---

### Q2: "What was the data leakage bug in the initial code, and how did you fix it?"
**Answer**:
> *"The original script called `.interpolate(method='time')` on the target column `nat_demand`. In time series, backward/forward interpolation uses future timestamps to estimate missing values in the past. When training on interpolated targets, the model implicitly learns information derived from future labels. I fixed this by keeping interpolation strictly for exogenous weather columns (where short missing sensor gaps can be filled), and applying `df.dropna(subset=['nat_demand'])` to the target. We never synthesize ground truth labels."*

---

### Q3: "Why did you replace $T^2$ with $(T - 24)^2$?"
**Answer**:
> *"In temperate climates, the load-temperature relationship is a U-curve because cold temperatures cause resistive heating demand while hot temperatures cause air conditioning demand, with a minimum around 18–22 °C. The previous author wrote `T**2`, claiming it modeled this U-curve. However, Panama is a tropical country where ambient temperatures typically stay between 20 °C and 35 °C. The function $T^2$ has its parabola vertex at 0 °C, meaning across Panama's entire temperature range it is strictly increasing with no minimum. To represent real thermodynamic cooling demand, I centered the quadratic term at 24 °C (standard tropical thermal comfort setpoint): $(T_{2M} - 24)^2$. This gives the model a physically meaningful curvature around human comfort."*

---

### Q4: "Why did you use XGBoost instead of an LSTM, GRU, or Temporal Fusion Transformer?"
**Answer**:
> *"There are three core reasons:
> 1. **Empirical Performance on Tabular Time Series**: Landmark papers (e.g., Grinsztajn et al., 2022; Shwartz-Ziv & Armon) demonstrate that tree-based models like XGBoost consistently outperform deep learning on tabular data with structured autoregressive features.
> 2. **Operational Latency & Reliability**: In a real grid energy management system (EMS), next-hour inference runs on tight cycles. XGBoost inference takes under 5 milliseconds with negligible CPU footprint, unlike deep learning which requires GPU infrastructure and runtime dependencies.
> 3. **Interpretability**: Grid dispatchers must trust why a load forecast shifted. XGBoost provides explicit split gain, coverage, and feature importances, allowing engineers to verify whether an anomaly was caused by temperature spikes or holiday calendar shifts."*

---

### Q5: "Why did you use same-hour-yesterday as your baseline instead of previous-hour ($t-1$)?"
**Answer**:
> *"While $t-1$ has high autocorrelation, power grids exhibit a dominant 24-hour diurnal cycle. At 7:00 AM, the previous hour is 6:00 AM (night trough before ramp-up), while yesterday at 7:00 AM represents the exact same morning industrial and residential start-up. In electrical engineering and grid operations, same-hour-yesterday persistence is the standard benchmark for short-term forecasting. Our model reduced the persistence error from 6.03% to 1.63% (a +4.39 percentage point improvement)."*

---

### Q6: "Why couldn't you use standard 5-Fold Cross Validation?"
**Answer**:
> *"Standard K-Fold randomly assigns records into folds, breaking the chronological arrow of time. If fold 1 contains random hours from 2018 and fold 2 contains random hours from 2016, training on fold 1 to test on fold 2 allows future information to predict the past. We used a strict chronological split: the first 80% (January 2015 to May 2019) for training, and the final 20% (May 2019 to June 2020) for out-of-sample testing. If cross-validation is needed, TimeSeriesSplit (expanding window walk-forward) must be used."*

---

### Q7: "Why did you include month features if you only wanted hour and day-of-week?"
**Answer**:
> *"I checked the dataset date range programmatically before deciding. The Panama continuous dataset spans from January 2015 to June 2020 (5.5 years). Because the data spans multiple full calendar cycles, annual seasonality (dry season vs. wet rainy season in Central America) repeats across years. Therefore, including `month_sin` and `month_cos` allowed the model to learn macro seasonal demand shifts without risk of single-year overfitting."*

---

### Q8: "How does the model handle holidays?"
**Answer**:
> *"The raw dataset included `Holiday_ID`, `holiday`, and `school` flags. In our feature selection, we chose to exclude them initially to establish a lean, uncorrupted meteorological and autoregressive baseline. In the 'Limitations & Honest Caveats' section of our report, we explicitly document that public holidays are a primary source of residual error (because holiday demand looks like a Sunday rather than a weekday), and integrating a vetted calendar schedule is the next logical iteration."*

---

### Q9: "Why did you remove the auto-detection of columns in the official forecast comparison?"
**Answer**:
> *"The original code used substring matching like `'forecast' in col.lower()`. This is brittle and failed silently if headers changed. I opened and verified the actual CSV headers (`weekly pre-dispatch forecast.csv` has `datetime` and `load_forecast`), hardcoded the verified schema, and added timestamp intersection logic to ensure comparison metrics are calculated only over overlapping hours."*

---

### Q10: "Explain the metrics: Why MAE, RMSE, and MAPE together?"
**Answer**:
> - **MAE (19.57 MW)**: Gives the linear average error magnitude in physical units (megawatts), directly interpretable by grid engineers.
> - **RMSE (26.40 MW)**: Squares errors before averaging, penalizing large outlier errors heavily. When RMSE is close to MAE (here 26.4 vs 19.6), it indicates error distribution is tight without catastrophic spike failures.
> - **MAPE (1.63%)**: Normalizes error relative to load magnitude, enabling cross-system benchmarking. Sub-2% MAPE is considered utility-grade worldwide.
> - **$R^2$ (0.9799)**: Confirms that 98% of total grid load variance is explained by our 11 features."*

---

### Q11: "What are the limitations of this project?"
**Answer**:
> *"Being honest about limitations demonstrates engineering maturity:
> 1. It is a single-step next-hour model, not a multi-step rolling autoregressive forecaster.
> 2. It does not account for holiday schedule overrides.
> 3. Meteorological data was collected from one primary weather station (Tocumen Airport, Panama City); incorporating multi-regional solar irradiance and humidity would improve localized storm response."*

---

### Q12: "How would you package and deploy this model in production?"
**Answer**:
> *"We containerized the application with Docker using `python:3.11-slim`, with `libgomp1` installed for OpenMP parallel tree building, `MPLBACKEND=Agg` for headless execution, and Docker Compose with volume-mounted input/output folders. In production, this container can run as a scheduled cron service or event-driven worker triggered by SCADA/EMS hourly data arrival, publishing forecasts back via a REST API or Kafka topic."*

---

## 7. Cheat Sheet: Numbers to Memorize

```
┌─────────────────────────────────────────────────────────────┐
│                       PROJECT METRICS                       │
├──────────────────────┬──────────────────────────────────────┤
│ Metric               │ Value                                │
├──────────────────────┼──────────────────────────────────────┤
│ Model MAPE           │ 1.63%                                │
│ Baseline MAPE        │ 6.03% (Same-hour-yesterday)          │
│ Lift                 │ +4.39 percentage points              │
│ MAE                  │ 19.57 MW                             │
│ RMSE                 │ 26.40 MW                             │
│ R² Score             │ 0.9799 (98% variance explained)      │
│ Average Load         │ 1,211.54 MW                          │
│ Dataset Span         │ 5.5 Years (2015-01-03 to 2020-06-27) │
│ Test Set Size        │ 9,576 hourly samples (Last 20%)      │
│ Number of Features   │ 11 explicit features                 │
│ Top Feature          │ load_lag_1h (42.2% gain)             │
│ Second Feature       │ load_lag_168h (20.4% gain)           │
│ Top Weather Feature  │ T2M_toc (13.5% gain)                 │
└──────────────────────┴──────────────────────────────────────┘
```