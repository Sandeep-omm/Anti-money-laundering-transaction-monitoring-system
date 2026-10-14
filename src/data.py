"""Dataset loading, normalization, and clearly marked demo data."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.sampling import sample_csv_rows

FEATURES = ["amount_paid", "amount_received", "amount_ratio", "bank_changed"]
DISPLAY_COLUMNS = ["timestamp", "from_bank", "from_account", "to_bank", "to_account", "amount_paid", "amount_received", "payment_currency", "receiving_currency", "payment_format", "is_laundering"]

ALIASES = {
    "timestamp": ("timestamp", "time", "date"),
    "from_bank": ("from_bank", "from bank", "origin_bank"),
    "from_account": ("from_account", "from account", "origin_account"),
    "to_bank": ("to_bank", "to bank", "destination_bank"),
    "to_account": ("to_account", "to account", "destination_account"),
    "amount_paid": ("amount_paid", "amount paid", "amount"),
    "amount_received": ("amount_received", "amount received"),
    "payment_currency": ("payment_currency", "payment currency", "currency"),
    "receiving_currency": ("receiving_currency", "receiving currency"),
    "payment_format": ("payment_format", "payment format", "payment_type", "payment type"),
    "is_laundering": ("is_laundering", "is laundering", "laundering", "label"),
}

def normalize_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Map common IBM AML column names into a stable application schema."""
    df = frame.copy()
    normalized = {str(c).strip().lower().replace("-", "_").replace(" ", "_"): c for c in df.columns}
    rename = {}
    for target, aliases in ALIASES.items():
        for alias in aliases:
            key = alias.lower().replace(" ", "_")
            if key in normalized:
                rename[normalized[key]] = target
                break
    df = df.rename(columns=rename)
    # IBM's transaction CSV has two columns both named Account. Pandas keeps
    # them as Account and Account.1, so map them by their source order.
    account_columns = [column for column in df.columns if str(column).strip().lower() == "account" or str(column).strip().lower().startswith("account.")]
    if account_columns:
        if "from_account" not in df:
            df = df.rename(columns={account_columns[0]: "from_account"})
        remaining = [column for column in account_columns if column in df.columns and column != "from_account"]
        if remaining and "to_account" not in df:
            df = df.rename(columns={remaining[0]: "to_account"})
    for col in DISPLAY_COLUMNS:
        if col not in df:
            df[col] = pd.NaT if col == "timestamp" else (0 if col in ("amount_paid", "amount_received", "is_laundering") else "Unknown")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ("amount_paid", "amount_received"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)
    df["is_laundering"] = pd.to_numeric(df["is_laundering"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    for col in ("from_bank", "from_account", "to_bank", "to_account", "payment_currency", "receiving_currency", "payment_format"):
        df[col] = df[col].fillna("Unknown").astype(str)
    ratio = df["amount_received"] / df["amount_paid"].replace(0, np.nan)
    df["amount_ratio"] = ratio.replace([np.inf, -np.inf], np.nan).fillna(1).clip(0, 100)
    df["bank_changed"] = (df["from_bank"] != df["to_bank"]).astype(int)
    return df

def load_transactions(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load a complete CSV (or its first N rows when nrows is specified)."""
    return normalize_transactions(pd.read_csv(path, nrows=nrows))

def load_transaction_sample(path: str | Path, nrows: int = 100_000, seed: int = 17) -> pd.DataFrame:
    """Load a bounded, file-wide sample for dashboard previews."""
    return normalize_transactions(sample_csv_rows(path, nrows=nrows, seed=seed))

def iter_transaction_chunks(path: str | Path, chunksize: int = 100_000):
    """Yield normalized transaction chunks without retaining the full CSV in RAM."""
    wanted = {"timestamp", "from_bank", "to_bank", "amount_paid", "amount_received", "payment_currency", "receiving_currency", "payment_format", "is_laundering"}
    def keep_column(column: str) -> bool:
        key = str(column).strip().lower().replace("-", "_").replace(" ", "_")
        return key in wanted or key in {"time", "date", "amount", "laundering", "label", "payment_type"}
    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=keep_column, dtype="string"):
        yield normalize_transactions(chunk)

def demo_transactions(n: int = 900, seed: int = 17) -> pd.DataFrame:
    """Generate illustrative, non-real transaction records for the dashboard."""
    rng = np.random.default_rng(seed)
    banks = np.array([f"BANK-{i:03d}" for i in range(1, 32)])
    formats = np.array(["Bank Transfer", "Cash", "Cheque", "Credit Card", "ACH", "Wire"])
    currencies = np.array(["USD", "EUR", "GBP", "INR", "CAD"])
    paid = np.round(np.exp(rng.normal(8.0, 1.25, n)), 2)
    suspicious = (rng.random(n) < 0.035).astype(int)
    paid[suspicious == 1] *= rng.uniform(2.2, 8, (suspicious == 1).sum())
    received = paid * rng.normal(0.98, 0.035, n)
    received[suspicious == 1] = paid[suspicious == 1] * rng.uniform(0.25, 1.5, (suspicious == 1).sum())
    frame = pd.DataFrame({
        "timestamp": pd.Timestamp.now().floor("h") - pd.to_timedelta(rng.integers(0, 30 * 24, n), unit="h"),
        "from_bank": rng.choice(banks, n), "from_account": rng.choice([f"ACCT-{i:05d}" for i in range(500)], n),
        "to_bank": rng.choice(banks, n), "to_account": rng.choice([f"ACCT-{i:05d}" for i in range(500)], n),
        "amount_paid": paid, "amount_received": np.round(received, 2),
        "payment_currency": rng.choice(currencies, n), "receiving_currency": rng.choice(currencies, n),
        "payment_format": rng.choice(formats, n, p=[.38, .13, .08, .12, .17, .12]),
        "is_laundering": suspicious,
    })
    return normalize_transactions(frame)
