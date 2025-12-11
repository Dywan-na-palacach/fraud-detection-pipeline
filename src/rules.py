import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .features import add_time_cols, add_numeric_stuff
from .load_data import load_transactions


def apply_simple_rules(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Kilka prostych reguł fraudowych. Raczej demo niż produkcja.
    """
    if df is None:
        df = load_transactions()

    df = add_time_cols(df)
    df = add_numeric_stuff(df)
    df = df.copy()

    high_dev = df["spending_deviation_score"] > 1.5
    high_vel = df["velocity_score"] > df["velocity_score"].quantile(0.95)
    high_geo = df["geo_anomaly_score"] > 2
    night_big = (df["is_night"] == 1) & (
        df["amount"] > df["amount"].quantile(0.9)
    )
    new_dev = df["is_new_device"] == 1

    rule_flag = high_dev | high_vel | high_geo | night_big | new_dev

    df["rule_is_fraud"] = rule_flag.astype(int)
    return df


def eval_rules(df: pd.DataFrame | None = None) -> None:
    if df is None:
        df = apply_simple_rules()
    elif "rule_is_fraud" not in df.columns:
        df = apply_simple_rules(df)

    y_true = df["is_fraud"].astype(int)
    y_pred = df["rule_is_fraud"].astype(int)

    print("\n=== rule-based metrics ===")
    print("confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nreport:")
    print(classification_report(y_true, y_pred, digits=4))
