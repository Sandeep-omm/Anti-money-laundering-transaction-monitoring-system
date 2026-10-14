"""Reader for the IBM AML laundering typology pattern text files."""
from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path

def summarize_patterns(path: str | Path, max_examples: int = 20) -> dict:
    """Count typology blocks and listed transaction records in a Patterns.txt file."""
    categories: Counter[str] = Counter()
    records = 0
    examples: list[dict] = []
    current = "Unspecified"
    with Path(path).open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if text.upper().startswith("BEGIN LAUNDERING ATTEMPT"):
                current = text.split("-", 1)[1].strip() if "-" in text else "Unspecified"
                categories[current] += 1
                continue
            if text.upper().startswith("END LAUNDERING ATTEMPT"):
                current = "Unspecified"
                continue
            row = next(csv.reader([text]))
            if len(row) < 11:
                continue
            records += 1
            if len(examples) < max_examples:
                examples.append({
                    "typology": current,
                    "timestamp": row[0], "from_bank": row[1], "from_account": row[2],
                    "to_bank": row[3], "to_account": row[4],
                    "amount_received": row[5], "receiving_currency": row[6],
                    "amount_paid": row[7], "payment_currency": row[8],
                    "payment_format": row[9], "is_laundering": row[10],
                })
    return {"categories": dict(categories), "pattern_blocks": sum(categories.values()), "listed_transactions": records, "examples": examples}
