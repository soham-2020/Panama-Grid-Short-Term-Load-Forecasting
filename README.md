# Panama Grid — Next-Hour Load Forecasting

**Task**: Predict `nat_demand` (MW) for the **next hour** using XGBoost on hourly Panama grid data.


---

## Dataset

| Field | Value |
|---|---|
| Source | Panama Case Study — `continuous_dataset.csv` |
| Granularity | Hourly |
| Date range | 2015-01-03 → 2020-06-27 (5.5 years, 48 048 records) |
| Target column | `nat_demand` (MW, national grid load) |
| Weather station | Tocumen Airport (`T2M_toc`) — Panama City |

---

## Features Used

The feature list below is exact and fixed — no dynamic column detection. All 11 features listed
were present in the run that produced the reported metrics.

| Feature | Description |
|---|---|
| `hour_sin`, `hour_cos` | Cyclical hour-of-day encoding |
| `dow_sin`, `dow_cos` | Cyclical day-of-week encoding |
| `month_sin`, `month_cos` | Cyclical month encoding — **included** (dataset spans 5.5 years > 1 year threshold) |
| `load_lag_1h` | `nat_demand` at t − 1 h |
| `load_lag_24h` | `nat_demand` at t − 24 h (same hour yesterday) |
| `load_lag_168h` | `nat_demand` at t − 168 h (same hour last week) |
| `T2M_toc` | 2-metre air temperature at Tocumen (°C) |
| `T_dev_sq` | `(T2M_toc − 24)²` — quadratic deviation from 24 °C thermal-comfort baseline |

Total: 11 features. The code includes `month_sin`/`month_cos` automatically when the dataset
spans more than one calendar year; at 5.5 years this condition is always met for this dataset.

---

## Preprocessing

- **Target gaps**: rows with missing `nat_demand` are **dropped** (not interpolated).
  Interpolating the forecasting target leaks future information into the training set.
- **Weather gaps**: short gaps in exogenous weather columns are filled with **linear interpolation**
  (equivalent to time-based interpolation on uniform hourly data).
- **Lag rows**: the first 168 rows that acquire NaN lags are dropped after lag construction.

---

## Validation Method

| Setting | Value |
|---|---|
| Split type | Chronological (no shuffling) |
| Train | First 80% of timesteps — 2015-01-10 → 2019-05-25 (38 304 h) |
| Test | Last 20% of timesteps — 2019-05-25 → 2020-06-27 (9 576 h) |
| Shuffle | None — prevents future-data leakage |

---

## Model

```python
XGBRegressor(
    n_estimators     = 300,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = 42,
    n_jobs           = -1,
)
```

---

## Results

*Measured on the chronological 20% test split (2019-05-25 → 2020-06-27, 9 576 h).*

| Metric | XGBoost | Naive baseline (same-hour-yesterday) |
|---|---|---|
| MAPE | **1.63 %** | 6.03 % |
| MAE | 19.57 MW | — |
| RMSE | 26.40 MW | — |
| R² | 0.9799 | — |

**XGBoost improvement over baseline: +4.39 pp MAPE**
*(XGBoost 1.63 % vs same-hour-yesterday persistence 6.03 %)*

---

## Top Feature Importances (Gain)

| Feature | Importance |
|---|---|
| `load_lag_1h` | 0.422 |
| `load_lag_168h` | 0.204 |
| `T2M_toc` | 0.135 |
| `load_lag_24h` | 0.103 |
| `dow_sin` | 0.049 |

---

## Official Forecast Comparison

The script optionally compares XGBoost against the official weekly pre-dispatch forecast.

**Column names verified directly from CSV header:**

| File | Datetime column | Forecast column |
|---|---|---|
| `weekly pre-dispatch forecast.csv` | `datetime` | `load_forecast` |

Comparison is restricted to the **intersection of timestamps** between the official forecast
file and the test set — no independent-window or full-file evaluation.

---

## Limitations & Honest Caveats

- Results reflect the single chronological test split described above — no cross-validation.
- No hyperparameter search was performed; the configuration is a defensible default.
- Holiday and school-calendar columns in the raw CSV were not used; including them may
  reduce errors around public holidays.
- No claims about real-time latency, edge deployment, or production readiness are made.

---

## Running the Script

```bash
# With explicit path:
python analysis.py path/to/continuous_dataset.csv

# With default convention (expects file at data/continuous_dataset.csv):
python analysis.py
```

Outputs saved to `outputs/`:

| File | Contents |
|---|---|
| `outputs/forecast_analysis.png` | Actual vs forecast + residuals + feature importance |
| `outputs/diurnal_curve.png` | Average hourly load — weekday vs weekend |
| `outputs/temperature_scatter.png` | Temperature vs demand (coloured by hour) |

---

## Installation

```bash
pip install pandas numpy xgboost scikit-learn matplotlib seaborn
```
---

## Running with Docker

### Option A: Using Docker CLI
```bash
# 1. Build image
docker build -t panama-grid .

# 2. Run with mounted data and outputs directories
docker run --rm \
  -v "${PWD}/data:/app/data:ro" \
  -v "${PWD}/outputs:/app/outputs" \
  panama-grid
```

### Option B: Using Docker Compose
```bash
docker compose up --build
```
The forecast results and plots will be saved directly into `./outputs/`.
