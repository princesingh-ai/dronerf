from pathlib import Path
import json

import pandas as pd

RESULTS_DIR = Path("docs/external_val_results")


def load_results() -> pd.DataFrame:
    """Load all external benchmark results."""

    records = []

    for file in RESULTS_DIR.glob("*.json"):

        with open(file) as f:
            benchmark = json.load(f)

        records.extend(benchmark["results"])

    return pd.DataFrame(records)