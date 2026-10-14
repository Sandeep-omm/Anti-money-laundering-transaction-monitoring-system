"""Train and save a baseline AML classifier from an IBM-style CSV."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.data import FEATURES, iter_transaction_chunks
from src.risk import _features

def train(input_path: str | Path, output_path: str | Path = "models/aml_model.joblib", chunksize: int = 100_000, negatives_per_chunk: int = 2_500) -> dict:
    """Stream the source file and train on all positives plus sampled negatives."""
    pieces = []
    total_rows = 0
    for index, chunk in enumerate(iter_transaction_chunks(input_path, chunksize=chunksize)):
        total_rows += len(chunk)
        positives = chunk[chunk["is_laundering"] == 1]
        negatives = chunk[chunk["is_laundering"] == 0]
        if len(positives) > 10_000:
            positives = positives.sample(n=10_000, random_state=42 + index)
        if len(negatives) > negatives_per_chunk:
            negatives = negatives.sample(n=negatives_per_chunk, random_state=42 + index)
        pieces.extend([part for part in (positives, negatives) if not part.empty])
    if not pieces:
        raise ValueError("No transaction rows were found in the input file.")
    df = pd.concat(pieces, ignore_index=True)
    if df["is_laundering"].nunique() < 2:
        raise ValueError("Training data must contain both laundering labels (0 and 1).")
    print(f"Read {total_rows:,} source rows; training on {len(df):,} rows (all positive rows up to a per-chunk cap plus sampled negatives).")
    x, y = _features(df), df["is_laundering"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42, stratify=y)
    positive_weight = max(1.0, float((y_train == 0).sum()) / max(1, int((y_train == 1).sum())))
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                        ("classifier", XGBClassifier(n_estimators=350, max_depth=5, learning_rate=.05,
                         subsample=.85, colsample_bytree=.85, reg_lambda=2, scale_pos_weight=positive_weight,
                         eval_metric="logloss", random_state=42, n_jobs=-1))])
    pipeline.fit(x_train, y_train)
    probs = pipeline.predict_proba(x_test)[:, 1]
    metrics = {"average_precision": float(average_precision_score(y_test, probs)), "roc_auc": float(roc_auc_score(y_test, probs))}
    destination = Path(output_path); destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": pipeline, "features": FEATURES, "metrics": metrics}, destination)
    print(f"Saved model to {destination.resolve()}")
    print(metrics)
    print(classification_report(y_test, probs >= .5, zero_division=0))
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="data/kaggle-full/HI-Large_Trans.csv", help="Input transaction CSV path")
    parser.add_argument("--output", default="models/aml_model.joblib")
    parser.add_argument("--chunksize", type=int, default=100_000, help="Rows read per chunk")
    parser.add_argument("--negatives-per-chunk", type=int, default=2_500, help="Normal transactions sampled per chunk")
    args = parser.parse_args()
    train(args.input, args.output, chunksize=args.chunksize, negatives_per_chunk=args.negatives_per_chunk)
