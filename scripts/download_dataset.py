"""Download and verify the IBM AML HI-Small transaction CSV."""
from __future__ import annotations
import hashlib
from pathlib import Path
import urllib.request
import zipfile

URL = "https://huggingface.co/datasets/eexzzm/IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans/resolve/main/HI-Small_Trans.csv.zip"
SHA256 = "b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040"
DESTINATION = Path(__file__).resolve().parents[1] / "data" / "HI-Small_Trans.csv"

def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    archive = DESTINATION.with_suffix(".csv.zip")
    if not DESTINATION.exists():
        print("Downloading HI-Small transaction data (~91 MB compressed)...")
        urllib.request.urlretrieve(URL, archive)
        with zipfile.ZipFile(archive) as zipped:
            member = next((name for name in zipped.namelist() if name.endswith("HI-Small_Trans.csv")), None)
            if member is None:
                raise RuntimeError("The downloaded archive does not contain HI-Small_Trans.csv")
            with zipped.open(member) as source, DESTINATION.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
        archive.unlink(missing_ok=True)
    checksum = hashlib.sha256()
    with DESTINATION.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    digest = checksum.hexdigest()
    if digest != SHA256:
        DESTINATION.unlink(missing_ok=True)
        raise RuntimeError(f"Dataset checksum mismatch: expected {SHA256}, received {digest}")
    print(f"Verified {DESTINATION} (SHA-256 {digest})")

if __name__ == "__main__":
    main()
