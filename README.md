# Panama Grid — Next-Hour Load Forecasting

Predict Panama's next-hour national electricity demand (nat_demand) with an XGBoost regressor, using a chronological train/test split and hourly weather/load features.

## Dataset and first-time setup

The project uses the preprocessed hourly Panama case-study data published by Ernesto Aguilar Madrid on Mendeley Data:

- **Dataset:** [Short-term electricity load forecasting (Panama case study)](https://data.mendeley.com/datasets/byx7sztj59/1)
- **DOI:** [10.17632/byx7sztj59.1](https://doi.org/10.17632/byx7sztj59.1)
- **License:** CC BY 4.0 (see the dataset record)
- **Coverage:** hourly records from January 2015 through June 2020, combining historical grid demand with weather and calendar data.

The Mendeley record describes a continuous CSV with the merged hourly records. The CSV is not committed here. To acquire it:

1. Open the Mendeley Data record above and select **Download All**.
2. Extract the downloaded archive.
3. Identify the preprocessed continuous dataset CSV (not the separate weekly pre-dispatch forecast CSV). It must have datetime and nat_demand columns; T2M_toc is the temperature feature used when present.
4. From the repository root, create data/ and copy/rename that CSV to data/continuous_dataset.csv.

For example, after extracting to dataset-download/:

```bash
mkdir -p data outputs
find dataset-download -type f -iname '*.csv' -print
# Copy the continuous merged CSV identified above:
cp "dataset-download/<continuous-dataset-file>.csv" data/continuous_dataset.csv
```

The dataset is attributed to its original creator under CC BY 4.0. Cite the dataset DOI when reusing it. Because the data is distributed separately, cloning this repository alone does not download it.

## Requirements and input checks

Install Python 3.11 or later, then:

```bash
python -m pip install -r requirements.txt
```

The script checks that the CSV exists and can be read; verifies the required datetime and nat_demand columns; parses timestamps; rejects duplicate timestamps and non-numeric feature values; and requires at least 200 rows for the 168-hour lag and temporal split. Missing target values are dropped. Optional weather columns are linearly interpolated; the model uses T2M_toc when supplied. Errors name the missing file or invalid column so setup issues are easier to diagnose.

## Run the forecast

From the repository root, after placing the dataset at data/continuous_dataset.csv:

```bash
python analysis.py
```

Or pass a CSV path explicitly:

```bash
python analysis.py path/to/continuous_dataset.csv
```

The script makes a chronological 80/20 split (no shuffle), trains XGBoost, reports metrics against the same-hour-yesterday baseline, and writes plots under outputs/. If data/weekly pre-dispatch forecast.csv is also present, it reports the optional comparison with the official pre-dispatch forecast; the model run does not require that second file.

## Synthetic-data smoke test

This check exercises missing-file and input-schema messages, then runs the complete forecasting and plotting path on a small generated dataset. It does not read the Panama dataset and its metrics are not project results.

```bash
python smoke_test.py
```

A successful run prints PASS and creates temporary plots which are removed when the test exits.

## Docker

Build and run after acquiring the dataset as described above:

```bash
mkdir -p data outputs
docker build -t panama-grid .
docker run --rm \
  -v "${PWD}/data:/app/data:ro" \
  -v "${PWD}/outputs:/app/outputs" \
  panama-grid
```

Or use Docker Compose:

```bash
mkdir -p data outputs
docker compose up --build
```

Both methods use data/continuous_dataset.csv and save plots to outputs/. To run the same synthetic smoke test inside the built image:

```bash
docker compose build
docker compose run --rm --entrypoint python panama-grid smoke_test.py
```

## Features and evaluation

The script creates cyclical encodings for hour and weekday, load lags at 1, 24, and 168 hours, and uses T2M_toc plus (T2M_toc - 24)^2 when that weather column exists. Month encodings are added for datasets spanning more than one year. The target is not interpolated to avoid using future information. Weather values are linearly interpolated before lag rows are removed.

The test split is the final 20% of the chronologically ordered records. Reported metrics are MAE, RMSE, MAPE, and R². The same-hour-yesterday baseline uses the 24-hour lag.

## Previously reported results — not reproduced in this update

The repository previously documented **1.63% MAPE** for XGBoost and **6.03% MAPE** for the same-hour-yesterday baseline on a 2019–2020 chronological test period. These values are retained as the project's prior reported results; they have **not** been independently reproduced as part of this reproducibility update. The synthetic smoke test only verifies that the pipeline runs and produces outputs. Running with the acquired source dataset is required to recalculate and verify model metrics.

## Outputs

- outputs/forecast_analysis.png — actual vs. forecast, residuals, and feature importance
- outputs/diurnal_curve.png — average weekday and weekend hourly demand
- outputs/temperature_scatter.png — temperature vs. demand when T2M_toc is available

The published dataset's source description, upstream data sources, and license are available on the [Mendeley Data record](https://data.mendeley.com/datasets/byx7sztj59/1).
