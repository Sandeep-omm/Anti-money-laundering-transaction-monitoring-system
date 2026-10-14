"""Risk scoring with optional trained model and transparent demo fallback."""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.data import FEATURES

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "aml_model.joblib"

def _features(frame: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=frame.index)
    x["amount_paid"] = np.log1p(pd.to_numeric(frame["amount_paid"], errors="coerce").fillna(0).clip(lower=0))
    x["amount_received"] = np.log1p(pd.to_numeric(frame["amount_received"], errors="coerce").fillna(0).clip(lower=0))
    x["amount_ratio"] = (pd.to_numeric(frame["amount_received"], errors="coerce").fillna(0) / pd.to_numeric(frame["amount_paid"], errors="coerce").replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1).clip(0, 100)
    x["bank_changed"] = (frame.get("from_bank", "") != frame.get("to_bank", "")).astype(int) if isinstance(frame.get("from_bank", ""), pd.Series) else 0
    return x[FEATURES]

def score_transactions(frame: pd.DataFrame, model_path: str | Path = MODEL_PATH) -> pd.DataFrame:
    result = frame.copy()
    x = _features(result)
    path = Path(model_path)
    if path.exists():
        bundle = joblib.load(path)
        model = bundle["model"] if isinstance(bundle, dict) else bundle
        probabilities = model.predict_proba(x)[:, 1]
        result["scoring_method"] = "Trained model"
    else:
        # Demo-only heuristic; never presented as a trained probability.
        amount = pd.to_numeric(result["amount_paid"], errors="coerce").fillna(0)
        ratio = x["amount_ratio"]
        raw = (amount >= 10000).astype(float) * .32 + (amount >= 50000).astype(float) * .23
        raw += ((ratio < .8) | (ratio > 1.2)).astype(float) * .28
        raw += (result["from_bank"].astype(str) != result["to_bank"].astype(str)).astype(float) * .08
        probabilities = np.clip(raw, 0, .99).to_numpy()
        result["scoring_method"] = "Illustrative rules (no model artifact)"
    result["risk_probability"] = probabilities
    result["risk_score"] = np.round(probabilities * 100).astype(int)
    result["risk_level"] = np.select([probabilities >= .65, probabilities >= .30], ["High Risk", "Medium Risk"], default="Low Risk")
    return result
