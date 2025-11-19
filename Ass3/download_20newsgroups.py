"""Download the 20 Newsgroups corpus and store it as a JSON file.

Run with:
    python download_20newsgroups.py
"""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.datasets import fetch_20newsgroups


def main() -> None:
    out_dir = Path(__file__).with_suffix("").parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading 20 Newsgroups dataset (subset='all')...")
    dataset = fetch_20newsgroups(subset="all")
    print(
        f"Fetched {len(dataset.data)} documents across "
        f"{len(dataset.target_names)} categories."
    )

    payload = {
        "target_names": dataset.target_names,
        "data": dataset.data,
        "target": dataset.target.tolist(),
    }

    out_file = out_dir / "20newsgroups_raw.json"
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    print(f"Saved dataset to {out_file}")


if __name__ == "__main__":
    main()

