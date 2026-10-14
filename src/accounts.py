"""Account reference-data helpers."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.sampling import sample_csv_rows

ACCOUNT_COLUMNS = ["Bank Name", "Bank ID", "Account Number", "Entity ID", "Entity Name"]

def load_account_sample(path: str | Path, nrows: int = 25_000) -> pd.DataFrame:
    sample = sample_csv_rows(path, nrows=nrows)
    for column in ACCOUNT_COLUMNS:
        if column not in sample:
            sample[column] = "Unknown"
    return sample[ACCOUNT_COLUMNS]
