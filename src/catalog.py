"""Discover the IBM AML dataset's HI/LI, Small/Medium/Large file families."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

VARIANT_ORDER = {"HI-Large": 0, "HI-Medium": 1, "HI-Small": 2, "LI-Large": 3, "LI-Medium": 4, "LI-Small": 5}
TRANSACTION_NAME = re.compile(r"^(HI|LI)-(Large|Medium|Small)_Trans\.csv$", re.IGNORECASE)

@dataclass(frozen=True)
class DatasetVariant:
    name: str
    transaction_path: Path
    accounts_path: Path | None
    patterns_path: Path | None

def discover_variants(data_root: str | Path = "data") -> list[DatasetVariant]:
    """Find variants in data/ and data/kaggle-full/, preferring full download copies."""
    root = Path(data_root)
    found: dict[str, DatasetVariant] = {}
    for search_root in (root / "kaggle-full", root):
        if not search_root.is_dir():
            continue
        for transaction_path in search_root.glob("*_Trans.csv"):
            match = TRANSACTION_NAME.match(transaction_path.name)
            if not match:
                continue
            name = f"{match.group(1).upper()}-{match.group(2).title()}"
            found.setdefault(name, DatasetVariant(
                name=name,
                transaction_path=transaction_path,
                accounts_path=search_root / f"{name}_accounts.csv",
                patterns_path=search_root / f"{name}_Patterns.txt",
            ))
    return sorted(found.values(), key=lambda item: VARIANT_ORDER[item.name])
