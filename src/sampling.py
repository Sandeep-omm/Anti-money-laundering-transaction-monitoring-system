"""Memory-bounded CSV preview sampling spread across the complete file."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd

def sample_csv_rows(path: str | Path, nrows: int = 100_000, seed: int = 17) -> pd.DataFrame:
    """Read a deterministic row sample from multiple byte regions of a CSV.

    This avoids the misleading first-N-rows preview for chronologically ordered
    AML files, while seeking to small regions instead of parsing a multi-GB file.
    IBM's published CSV fields do not contain embedded newlines.
    """
    source = Path(path)
    nrows = max(1, int(nrows))
    size = source.stat().st_size
    rng = np.random.default_rng(seed)
    with source.open("rb") as handle:
        header = handle.readline()
        if not header:
            return pd.DataFrame()
        data_start = handle.tell()
        data_size = max(0, size - data_start)
        if data_size == 0:
            return pd.read_csv(BytesIO(header), dtype="string")
        regions = min(40, max(1, int(np.ceil(nrows / 5_000))))
        base, remainder = divmod(nrows, regions)
        pieces: list[pd.DataFrame] = []
        for region in range(regions):
            quota = base + (1 if region < remainder else 0)
            if quota <= 0:
                continue
            lower = data_start + int(data_size * region / regions)
            upper = data_start + int(data_size * (region + 1) / regions)
            if region == 0:
                offset = data_start
            else:
                offset = int(rng.integers(lower, max(lower + 1, upper)))
            handle.seek(offset)
            if offset > data_start:
                handle.readline()  # discard the partial record at this byte offset
            rows = [handle.readline() for _ in range(quota)]
            rows = [row for row in rows if row.strip()]
            if rows:
                pieces.append(pd.read_csv(BytesIO(header + b"".join(rows)), dtype="string"))
        if not pieces:
            return pd.read_csv(source, nrows=nrows, dtype="string")
        sample = pd.concat(pieces, ignore_index=True)
        if len(sample) > nrows:
            sample = sample.sample(n=nrows, random_state=seed).reset_index(drop=True)
        return sample
