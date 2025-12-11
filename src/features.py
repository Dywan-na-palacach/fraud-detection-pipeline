import numpy as np
import pandas as pd


BASE_CAT_COLS = [
    "transaction_type",
    "merchant_category",
    "location",
    "device_used",
    "payment_channel",
    "fraud_type",
]


def add_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" in df.columns:
        df["tx_hour"] = df["timestamp"].dt.hour
        df["tx_dow"] = df["timestamp"].dt.dayofweek
    else:
        df["tx_hour"] = 0
        df["tx_dow"] = 0
    return df


def add_numeric_stuff(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["log_amount"] = np.log1p(df["amount"])

    # noc ~ 0-5
    df["is_night"] = df["tx_hour"].between(0, 5).astype(int)

    if {"sender_account", "device_hash"}.issubset(df.columns):
        # sortujemy po kliencie + czasie, ale NIE zmieniamy sensownie indexu
        df = df.sort_values(["sender_account", "timestamp"])

        # transform zamiast apply => zawsze zwraca Series z takim samym indexem jak df
        df["is_new_device"] = (
            df.groupby("sender_account")["device_hash"]
            .transform(lambda s: ~s.duplicated())
            .astype(int)
        )
    else:
        df["is_new_device"] = 0


    for col in [
        "time_since_last_transaction",
        "spending_deviation_score",
        "velocity_score",
        "geo_anomaly_score",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_amount_deviation_feature(
    df: pd.DataFrame,
    account_col: str = "sender_account",
    min_tx_for_flag: int = 10,
    shrink_k: int = 20,
) -> pd.DataFrame:
    """
    Dodaje feature amount_dev_robust: ile "odchylona" jest kwota transakcji
    względem typowego zachowania klienta (median + MAD, ze shrinkage'em do globalnych statystyk).

    - account_col: kolumna z identyfikatorem konta/klienta (tu: sender_account)
    - min_tx_for_flag: próg dla flagi low_history_flag (mało historii na koncie)
    - shrink_k: parametr siły shrinkage'u między lokalnymi a globalnymi statystykami.
    """
    df = df.copy()

    if "amount" not in df.columns:
        return df

    # pracujemy na log_amount (upewniamy się, że jest)
    if "log_amount" not in df.columns:
        df["log_amount"] = np.log1p(df["amount"].astype(float))

    # liczba transakcji per konto
    if account_col in df.columns:
        tx_counts = df.groupby(account_col)["log_amount"].transform("count")
    else:
        # brak kolumny klienta -> nie ma sensu liczyć deviation per klient
        df["amount_dev_robust"] = 0.0
        df["low_history_flag"] = 1
        return df

    df["tx_count_per_account"] = tx_counts

    # lokalne mediany i MAD per konto
    def mad_series(s: pd.Series) -> float:
        med = np.median(s)
        return np.median(np.abs(s - med))

    grp = df.groupby(account_col)["log_amount"]
    med_local = grp.transform("median")
    mad_local = grp.transform(mad_series)

    # globalne median i MAD
    global_med = df["log_amount"].median()
    global_mad = mad_series(df["log_amount"])

    # współczynnik shrinkage'u: im więcej transakcji, tym bardziej ufamy lokalnym statystykom
    w = tx_counts / (tx_counts + shrink_k)

    med_shrunk = w * med_local + (1 - w) * global_med
    mad_shrunk = w * mad_local + (1 - w) * global_mad

    # żeby uniknąć dzielenia przez 0
    mad_shrunk = mad_shrunk.replace(0, global_mad)

    # robust deviation score (coś jak z-score, ale odporny na outliery)
    df["amount_dev_robust"] = (
        (df["log_amount"] - med_shrunk) / (1.4826 * mad_shrunk)
    )

    # flaga "mało historii"
    df["low_history_flag"] = (tx_counts < min_tx_for_flag).astype(int)

    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cats = [c for c in BASE_CAT_COLS if c in df.columns]
    if not cats:
        return df
    return pd.get_dummies(df, columns=cats, drop_first=True)

"""
old wersja bez docstring

def add_amount_deviation_feature(
    df: pd.DataFrame,
    account_col: str = "sender_account",
    min_tx_for_flag: int = 10,
    shrink_k: int = 20,
) -> pd.DataFrame:
    df = df.copy()

    # pracujemy na log_amount (upewnij się, że już jest policzone)
    if "log_amount" not in df.columns:
        df["log_amount"] = np.log1p(df["amount"])

    # liczba transakcji per konto
    tx_counts = df.groupby(account_col)["log_amount"].transform("count")
    df["tx_count_per_account"] = tx_counts

    # median i MAD per konto
    def mad_series(s: pd.Series) -> float:
        med = np.median(s)
        return np.median(np.abs(s - med))

    grp = df.groupby(account_col)["log_amount"]
    med_local = grp.transform("median")
    mad_local = grp.transform(mad_series)

    # globalne median i MAD
    global_med = df["log_amount"].median()
    global_mad = mad_series(df["log_amount"])

    # współczynnik shrinkage'u
    w = tx_counts / (tx_counts + shrink_k)

    med_shrunk = w * med_local + (1 - w) * global_med
    mad_shrunk = w * mad_local + (1 - w) * global_mad

    # unikamy MAD=0
    mad_shrunk = mad_shrunk.replace(0, global_mad)

    # robust deviation score
    df["amount_dev_robust"] = (
        (df["log_amount"] - med_shrunk) / (1.4826 * mad_shrunk)
    )

    # flaga "mało historii"
    df["low_history_flag"] = (tx_counts < min_tx_for_flag).astype(int)

    return df
"""

def add_merchant_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tworzy merchant_risk_score jako fraud_rate per merchant_category.
    Skalowane 0-1. Działa na pełnych danych, nie tylko train.
    """

    df = df.copy()
    if "merchant_category" not in df.columns or "is_fraud" not in df.columns:
        df["merchant_risk_score"] = 0.0
        return df

    # Fraud rate per merchant
    risk_table = (
        df.groupby("merchant_category")["is_fraud"]
        .mean()
        .rename("fraud_rate")
        .reset_index()
    )

    # Normalizacja → 0-1 (opcjonalne, ale ładnie wygląda w modelu + wykresie)
    min_r, max_r = risk_table["fraud_rate"].min(), risk_table["fraud_rate"].max()
    risk_table["merchant_risk_score"] = (risk_table["fraud_rate"] - min_r) / (max_r - min_r)

    df = df.merge(risk_table[["merchant_category", "merchant_risk_score"]], on="merchant_category", how="left")

    return df




def build_X_y(df: pd.DataFrame):
    """
    Prosty pipeline -> X, y, nazwy featurów.
    """
    df = df.copy()

    # minimalny sanity-check, żeby się nie bawić w KeyError
    required_cols = ["amount", "is_fraud"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Brakuje kolumn w danych: {missing}")

    # kolejność ma znaczenie: najpierw czasowe/numeryczne, potem deviation, na końcu one-hot
    df = add_time_cols(df)
    df = add_numeric_stuff(df)
    df = add_amount_deviation_feature(df)
    df = add_merchant_risk_score(df)
    df = one_hot_encode(df)

    y = df["is_fraud"].astype(int)

    base_feats = [
        "log_amount",
        "time_since_last_transaction",
        "spending_deviation_score",
        "velocity_score",
        "geo_anomaly_score",
        "tx_hour",
        "tx_dow",
        "is_night",
        "is_new_device",
        "amount_dev_robust",
        "low_history_flag",
        "merchant_risk_score",
    ]

    extra = [
        c
        for c in df.columns
        if c.startswith("transaction_type_")
        or c.startswith("merchant_category_")
        or c.startswith("location_")
        or c.startswith("device_used_")
        or c.startswith("payment_channel_")
    ]

    feat_cols = [c for c in base_feats + extra if c in df.columns]

    X = df[feat_cols].fillna(0.0)
    return X, y, feat_cols



