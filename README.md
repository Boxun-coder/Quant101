# Quant101

Statistical arbitrage research pipeline for S&P 500 equities using daily CRSP-style data from January 2016 through December 2024.

## Overview

This project builds a market-neutral stat arb workflow from raw equity data to a finished PDF report:

- Cleans and aligns CRSP-style daily data into panel matrices
- Constructs reversal, volume-shock, and 12M-1M momentum factors
- Beta-neutralizes the combined alpha and scales to gross leverage 2.0
- Runs a T+1 backtest with 5 bps transaction costs
- Generates a professional multi-page research report in PDF form

The implementation is designed to avoid look-ahead bias by shifting weights forward one trading day before applying returns.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── run_pipeline.py
└── src/
    ├── alphas.py
    ├── backtester.py
    ├── config.py
    ├── data_pipeline.py
    ├── portfolio.py
    ├── report_generator.py
    └── risk_model.py
```

## Data Expectations

Place the raw dataset at:

```text
data/raw/SP500_to202412.csv
```

Expected columns:

- `PERMNO`
- `date`
- `PRC`
- `VOL`
- `RET`
- `RETX`
- `SHROUT`
- `sprtrn`

The raw dataset is intentionally excluded from version control.

## Installation

```bash
python3 -m pip install -r requirements.txt
```

## Run The Pipeline

```bash
python3 run_pipeline.py
```

This will:

1. Read and clean the raw CRSP-style input
2. Save clean matrices to `data/clean/`
3. Build the combined raw alpha
4. Construct market-neutral target weights
5. Run the backtest and print a tear sheet
6. Save `results_matrix.parquet`
7. Generate `Final_Report.pdf`

## Core Research Choices

- Universe: S&P 500 constituents in the provided CRSP-style dataset
- Execution assumption: signals from Day T are executed on Day T+1
- Transaction costs: 5 bps per unit of turnover
- Neutralization: rolling 60-day beta neutralization to `sprtrn`
- Leverage target: gross exposure fixed at 2.0

## Main Outputs

Generated artifacts are written locally and ignored by Git:

- `data/clean/prc.parquet`
- `data/clean/ret.parquet`
- `data/clean/vol.parquet`
- `data/clean/mktcap.parquet`
- `data/clean/sprtrn.parquet`
- `data/clean/raw_alpha.parquet`
- `data/clean/target_weights.parquet`
- `data/clean/results_matrix.parquet`
- `Final_Report.pdf`

## Notes

- `report_generator.py` prefers `weasyprint` for HTML-to-PDF rendering when its native libraries are available.
- In environments where those native libraries are missing, the project falls back to a pure-Python PDF generation path so report creation can still complete.

## License

Add a license here if you want to publish the repository more broadly.
