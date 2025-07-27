from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from . import config


def _find_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_dataset(path: Path) -> Tuple[pd.Series, pd.Series]:
    """
    Loads the dataset and returns (text_series, label_series).

    It tries to:
    - Find a text column among TEXT_COLUMNS_CANDIDATES.
    - Optionally prepend title/headline if available.
    - Find a label column among LABEL_COLUMN_CANDIDATES.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)

    # detect columns
    text_col = _find_first_existing(df, config.TEXT_COLUMNS_CANDIDATES)
    title_col = _find_first_existing(df, config.TITLE_COLUMNS_CANDIDATES)
    label_col = _find_first_existing(df, config.LABEL_COLUMN_CANDIDATES)

    if text_col is None:
        raise ValueError(
            f"Could not find any text column in {config.TEXT_COLUMNS_CANDIDATES}. "
            f"Columns present: {list(df.columns)}"
        )

    if label_col is None:
        raise ValueError(
            f"Could not find any label column in {config.LABEL_COLUMN_CANDIDATES}. "
            f"Columns present: {list(df.columns)}"
        )

    # Combine title + text if title exists
    if title_col:
        text_series = (df[title_col].fillna("") + " " + df[text_col].fillna("")).astype(str)
    else:
        text_series = df[text_col].fillna("").astype(str)

    labels = df[label_col]

    return text_series, labels
