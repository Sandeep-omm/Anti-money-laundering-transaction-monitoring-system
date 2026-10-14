# ClearWatch — AML Transaction Monitoring

A local transaction monitoring dashboard, model training pipeline, and scoring API built with Python, Streamlit, XGBoost, Pandas, and FastAPI. Scores are investigation signals for human review; they are not findings of criminal activity.

## Features

- Interactive dashboard for transaction volume, observed dataset labels, payment formats, risk distribution, and a sortable review queue.
- CSV upload and local-file support for IBM AML-style column names.
- Individual transaction scoring and filtered CSV export.
- XGBoost training script with class-imbalance weighting and holdout metrics.
- FastAPI `/score` and `/health` endpoints.
- Explicit illustrative rule scoring when no trained model artifact is present.

## Project structure

```text
├── app.py                 # Streamlit dashboard
├── src/
│   ├── catalog.py         # HI/LI variant discovery and companion paths
│   ├── sampling.py        # Memory-bounded samples spread across large CSVs
│   ├── accounts.py        # Account reference-file reader
│   ├── patterns.py        # Typology pattern-file parser
│   ├── data.py            # Transaction schema and chunk readers
│   ├── risk.py            # Model or illustrative fallback scoring
│   ├── training.py        # XGBoost training and evaluation
│   └── main.py            # FastAPI service
├── data/                  # Local transaction CSVs (git-ignored)
├── models/                # Saved model artifacts (git-ignored)
├── requirements.txt
└── README.md
```

## Install

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Launch the dashboard

```bash
python -m streamlit run app.py
```

When the Kaggle collection is present, the dashboard discovers and lets you select all six transaction variants: HI/LI × Small/Medium/Large. HI-Large is selected by default. It also loads the matching accounts reference and patterns files when available. For very large transaction CSVs, the dashboard takes a deterministic sample from multiple regions of the file (up to 250,000 rows), so its preview does not only reflect the first chronological records and does not load a 17 GB CSV into memory. The Accounts tab uses a file-wide preview; the Patterns tab summarizes the complete, much smaller typology file. The dashboard's transaction KPIs and risk queue are sample-based, not full-file aggregates. You can upload a smaller CSV or enter a different local transaction path. If no data is present, ClearWatch displays illustrative demo records.

## Train a model

Select any downloaded transaction variant as the training input. For example, to train from the HI-Large file, run:

```bash
python -m src.training data/kaggle-full/HI-Large_Trans.csv
```

Training reads the transaction CSV in chunks, keeps laundering-positive records (with a per-chunk cap), and samples normal transactions to bound memory. Use `--chunksize` or `--negatives-per-chunk` to tune that tradeoff. It still scans the complete source file, so training on HI-Large can take a long time and requires enough disk and CPU throughput.

Optional artifact destination:

```bash
python -m src.training data/transactions.csv --output models/aml_model.joblib
```

The CSV needs a target column such as `Is Laundering` and transaction amounts. Common IBM headers such as `Amount Paid`, `Amount Received`, `From Bank`, `To Bank`, `Payment Currency`, and `Payment Format` are normalized automatically. Training uses amount paid, amount received, their ratio, and whether the bank identifiers differ. It saves a fitted preprocessing and XGBoost pipeline plus holdout average precision and ROC AUC. The available features are intentionally limited; meaningful deployment requires feature engineering, temporal and entity-level validation, leakage review, and threshold calibration against institution-specific outcomes.

After a model is saved, the dashboard and API load it automatically. A saved model's probability is a model score, not a calibrated estimate of criminality.

## Run the API

```bash
uvicorn src.main:app --reload
```

- `GET /health` — process health
- `POST /score` — returns score, risk band, scoring method, and review signal

Example request:

```json
{
  "amount_paid": 12000,
  "amount_received": 11800,
  "from_bank": "BANK-001",
  "to_bank": "BANK-014"
}
```

Open `/docs` on the local API server to inspect the interactive schema.

## Dataset

This workspace contains all 18 files in the Kaggle collection under `data/kaggle-full/`, including transaction, account, and typology pattern files for each HI/LI size. The included `data/HI-Small_Trans.csv` is a smaller fallback. Dataset files are excluded from Git because the collection is about 41.6 GB uncompressed. To download only HI-Small in a fresh checkout, run `python scripts/download_dataset.py`; for the entire Kaggle collection, use the Kaggle API after authenticating:

```powershell
python -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); a.dataset_download_files('ealtman2019/ibm-transactions-for-anti-money-laundering-aml', path='data/kaggle-full', unzip=True)"
```

IBM states that the records are synthetic and distributed under CDLA-Sharing-1.0; see [IBM's data repository](https://github.com/IBM/AML-Data), the [Kaggle dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml), and the [data license](https://spdx.org/licenses/CDLA-Sharing-1.0.html). Dataset labels and model outputs should be assessed for quality, bias, and suitability before operational use.

## Limitations

This is a demonstration and analyst-support starting point. The fallback scorer is a transparent heuristic, not a trained classifier. The provided feature set does not capture transaction networks, customer profiles, velocity, or temporal patterns, and the training split is a basic stratified random holdout rather than a production validation design. Do not use the demo output as the sole basis for customer action or regulatory reporting.
