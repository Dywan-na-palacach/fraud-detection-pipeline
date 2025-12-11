from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import DATA_PATH, DEBUG_SMALL_DATA, DEBUG_NROWS


def load_transactions(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """
    Wczytanie danych z csv i podstawowe ogarnięcie typów.
    W trybie DEBUG_SMALL_DATA wczytuje tylko pierwsze N wierszy.
    """
    path = Path(path)

    if DEBUG_SMALL_DATA:
        df = pd.read_csv(path, nrows=DEBUG_NROWS)
    else:
        df = pd.read_csv(path)

    # na wszelki wypadek
    df.columns = [c.strip() for c in df.columns]

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # is_fraud -> 0/1
    if "is_fraud" in df.columns:
        col = df["is_fraud"]
        if col.dtype == bool:
            df["is_fraud"] = col.astype(int)
        else:
            df["is_fraud"] = (
                col.astype(str)
                .str.lower()
                .map({"true": 1, "false": 0, "1": 1, "0": 0})
                .fillna(0)
                .astype(int)
            )

    return df
