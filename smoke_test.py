"""Exercise the forecasting pipeline with a small synthetic hourly dataset."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().with_name("analysis.py")


def run_analysis(csv_path: Path, workdir: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(csv_path)],
        cwd=workdir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="panama-forecast-smoke-") as temp_dir:
        root = Path(temp_dir)
        timestamps = pd.date_range("2020-01-01", periods=600, freq="h")
        hour = np.arange(len(timestamps))
        # Smooth, strictly positive synthetic demand and a simple weather signal.
        data = pd.DataFrame(
            {
                "datetime": timestamps,
                "nat_demand": 1000 + 85 * np.sin(2 * np.pi * hour / 24)
                + 20 * np.sin(2 * np.pi * hour / (24 * 7)),
                "T2M_toc": 27 + 3 * np.sin(2 * np.pi * hour / 24),
            }
        )
        valid_csv = root / "synthetic.csv"
        data.to_csv(valid_csv, index=False)

        # Check missing-file handling before the positive end-to-end run.
        missing = run_analysis(root / "does-not-exist.csv", root)
        if missing.returncode == 0 or "Dataset not found" not in missing.stderr:
            print("FAIL: missing-file validation did not return a clear error.")
            print(missing.stdout, missing.stderr)
            return 1

        # Check a CSV with the wrong columns receives a schema-specific error.
        bad_schema = root / "bad-schema.csv"
        pd.DataFrame({"time": timestamps, "load": data["nat_demand"]}).to_csv(
            bad_schema, index=False
        )
        invalid = run_analysis(bad_schema, root)
        if invalid.returncode == 0 or "missing required column(s)" not in invalid.stderr:
            print("FAIL: input-schema validation did not return a clear error.")
            print(invalid.stdout, invalid.stderr)
            return 1

        # Run the actual model/evaluation/plot path with synthetic input.
        result = run_analysis(valid_csv, root)
        if result.returncode != 0:
            print("FAIL: forecasting pipeline failed on synthetic data.")
            print(result.stdout)
            print(result.stderr)
            return result.returncode or 1

        expected_outputs = [
            root / "outputs" / "forecast_analysis.png",
            root / "outputs" / "diurnal_curve.png",
            root / "outputs" / "temperature_scatter.png",
        ]
        missing_outputs = [str(path.relative_to(root)) for path in expected_outputs if not path.is_file()]
        if missing_outputs:
            print(f"FAIL: expected output file(s) were not created: {', '.join(missing_outputs)}")
            print(result.stdout)
            return 1

        print("PASS: missing-file and schema checks passed; synthetic forecasting run completed.")
        print("Synthetic-data metrics are a pipeline smoke check, not project performance results.")
        print("Created: " + ", ".join(path.relative_to(root).as_posix() for path in expected_outputs))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
