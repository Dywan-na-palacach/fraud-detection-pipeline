from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import FIGURES_DIR


def basic_stats(df: pd.DataFrame) -> None:
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print("\nFraud distribution:")
    print(df["is_fraud"].value_counts())
    print("Fraud rate:", df["is_fraud"].mean())

    print("\nAmount stats:")
    print(df["amount"].describe())


def plot_class_balance(df: pd.DataFrame) -> None:
    counts = df["is_fraud"].value_counts().sort_index()
    plt.figure()
    counts.plot(kind="bar")
    plt.xticks([0, 1], ["non-fraud", "fraud"], rotation=0)
    plt.title("Class balance")
    plt.ylabel("Count")
    plt.tight_layout()
    out_path = FIGURES_DIR / "class_balance.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_hourly_fraud_rate(df: pd.DataFrame) -> None:
    if "tx_hour" not in df.columns:
        df = df.copy()
        df["tx_hour"] = df["timestamp"].dt.hour

    stats = df.groupby("tx_hour")["is_fraud"].mean()
    plt.figure()
    stats.plot(kind="bar")
    plt.title("Fraud rate by hour of day")
    plt.xlabel("Hour")
    plt.ylabel("Fraud rate")
    plt.tight_layout()
    out_path = FIGURES_DIR / "fraud_rate_by_hour.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def fraud_rate_by_column(df: pd.DataFrame, col: str, top_n: int = 10) -> pd.DataFrame:
    stats = (
        df.groupby(col)["is_fraud"]
        .agg(["count", "sum", "mean"])
        .rename(columns={"count": "n_tx", "sum": "n_fraud", "mean": "fraud_rate"})
        .sort_values("fraud_rate", ascending=False)
    )
    print(f"\nFraud rate by {col}:")
    print(stats.head(top_n))
    return stats


def plot_fraud_vs_log_amount(df: pd.DataFrame, n_bins: int = 30) -> None:
    """
    Fraud rate w zależności od log(amount).
    Dobre do zobaczenia, czy wysokie kwoty są bardziej ryzykowne.
    """
    df = df.copy()
    if "amount" not in df.columns or "is_fraud" not in df.columns:
        return

    df["log_amount"] = np.log1p(df["amount"])

    # binujemy log_amount na kwantylach, żeby rozkład był równomierny
    try:
        bins = pd.qcut(df["log_amount"], q=n_bins, duplicates="drop")
    except ValueError:
        # za mało danych -> odpuszczamy
        return

    grouped = df.groupby(bins)["is_fraud"].mean()

    plt.figure(figsize=(10, 5))
    grouped.plot(marker="o")
    plt.title("Fraud rate vs transaction amount (log-binned)")
    plt.xlabel("log(amount) quantile bin")
    plt.ylabel("Fraud rate")
    plt.grid(alpha=0.3)
    out_path = FIGURES_DIR / "fraud_vs_log_amount.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_amount_hist_overlay(df: pd.DataFrame, n_bins: int = 60) -> None:
    """
    Histogram kwot dla fraud vs non-fraud na jednym wykresie (log-scale na osi X).
    Pozwala zobaczyć różnice w rozkładzie.
    """
    df = df.copy()
    if "amount" not in df.columns or "is_fraud" not in df.columns:
        return

    legit = df.loc[df["is_fraud"] == 0, "amount"]
    fraud = df.loc[df["is_fraud"] == 1, "amount"]

    # żeby nie rozwaliły wykresu pojedyncze outliery
    legit = legit[legit > 0]
    fraud = fraud[fraud > 0]

    plt.figure(figsize=(10, 5))
    plt.hist(
        np.log1p(legit),
        bins=n_bins,
        alpha=0.5,
        density=True,
        label="non-fraud",
    )
    plt.hist(
        np.log1p(fraud),
        bins=n_bins,
        alpha=0.5,
        density=True,
        label="fraud",
    )
    plt.title("Distribution of log(amount) for fraud vs non-fraud")
    plt.xlabel("log(amount)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    out_path = FIGURES_DIR / "amount_hist_overlay.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_fraud_vs_amount_deviation(df: pd.DataFrame, n_bins: int = 25) -> None:
    """
    Fraud rate w zależności od amount_dev_robust (feature 'nietypowości').
    Zakładamy, że kolumna amount_dev_robust jest już policzona w features.py.
    """
    df = df.copy()
    if "amount_dev_robust" not in df.columns or "is_fraud" not in df.columns:
        print("Brak kolumny 'amount_dev_robust' – pomijam wykres fraud_vs_amount_dev.")
        return

    dev = df["amount_dev_robust"].clip(-10, 10)  # przycinamy ekstremy dla czytelności

    try:
        bins = pd.qcut(dev, q=n_bins, duplicates="drop")
    except ValueError:
        return

    grouped = df.groupby(bins)["is_fraud"].mean()

    plt.figure(figsize=(10, 5))
    grouped.plot(marker="o")
    plt.title("Fraud rate vs amount deviation (robust)")
    plt.xlabel("amount_dev_robust quantile bin")
    plt.ylabel("Fraud rate")
    plt.grid(alpha=0.3)
    out_path = FIGURES_DIR / "fraud_vs_amount_deviation.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_boxplots_for_features(
    df: pd.DataFrame,
    cols: List[str],
) -> None:
    """
    Boxploty dla wybranych cech numerycznych, rozdzielone na fraud / non-fraud.
    """
    df = df.copy()
    if "is_fraud" not in df.columns:
        return

    for col in cols:
        if col not in df.columns:
            continue

        plt.figure(figsize=(6, 5))
        data = [df.loc[df["is_fraud"] == 0, col], df.loc[df["is_fraud"] == 1, col]]
        plt.boxplot(data, labels=["non-fraud", "fraud"], showfliers=False)
        plt.title(f"{col} distribution by class (without outliers)")
        plt.ylabel(col)
        out_path = FIGURES_DIR / f"boxplot_{col}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved: {out_path}")


def plot_corr_heatmap(df: pd.DataFrame) -> None:
    """
    Prosta heatmapa korelacji cech numerycznych.
    """
    num_df = df.select_dtypes(include=["number"])
    if num_df.empty:
        return

    corr = num_df.corr()

    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(
        ticks=np.arange(len(corr.columns)),
        labels=corr.columns,
        rotation=90,
        fontsize=6,
    )
    plt.yticks(
        ticks=np.arange(len(corr.columns)),
        labels=corr.columns,
        fontsize=6,
    )
    plt.title("Correlation heatmap (numeric features)")
    plt.tight_layout()
    out_path = FIGURES_DIR / "correlation_heatmap.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_pareto_merchant_category(df: pd.DataFrame, top_n: int = 20) -> None:
    """
    Prosty wykres Pareto dla merchant_category:
    ile fraudów przypada na top k kategorii.
    """
    df = df.copy()
    if "merchant_category" not in df.columns or "is_fraud" not in df.columns:
        return

    fraud_counts = (
        df.loc[df["is_fraud"] == 1]
        .groupby("merchant_category")["is_fraud"]
        .count()
        .sort_values(ascending=False)
    )

    if fraud_counts.empty:
        return

    fraud_counts = fraud_counts.head(top_n)
    cum_frac = fraud_counts.cumsum() / fraud_counts.sum()

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(fraud_counts.index, fraud_counts.values)
    ax1.set_xticklabels(fraud_counts.index, rotation=45, ha="right")
    ax1.set_ylabel("Number of frauds")
    ax1.set_title("Pareto of frauds by merchant_category (top categories)")

    ax2 = ax1.twinx()
    ax2.plot(fraud_counts.index, cum_frac.values, marker="o", color="tab:red")
    ax2.set_ylabel("Cumulative fraction of frauds")

    plt.tight_layout()
    out_path = FIGURES_DIR / "pareto_merchant_category.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_merchant_risk(df: pd.DataFrame) -> None:
    """
    Ranking kategorii merchant_category po fraud_rate (0-1).
    To jest nasz merchant_risk_score z perspektywy analityka.
    """
    if "merchant_category" not in df.columns or "is_fraud" not in df.columns:
        print("Brak merchant_category lub is_fraud – pomijam merchant risk plot.")
        return

    grouped = (
        df.groupby("merchant_category")["is_fraud"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))
    grouped.plot(kind="bar")
    plt.title("Fraud rate by merchant_category (merchant risk score)")
    plt.ylabel("Fraud rate")
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.3, axis="y")
    out_path = FIGURES_DIR / "merchant_risk_score.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")
