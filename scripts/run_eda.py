from src.load_data import load_transactions
from src.eda import basic_stats, plot_class_balance, plot_hourly_fraud_rate, fraud_rate_by_column
from src.features import add_time_cols



def main():
    df = load_transactions()
    df = add_time_cols(df)

    basic_stats(df)
    plot_class_balance(df)
    plot_hourly_fraud_rate(df)

    # segmentacje
    for col in ["payment_channel", "merchant_category", "location", "device_used"]:
        if col in df.columns:
            fraud_rate_by_column(df, col, top_n=10)


def plot_merchant_risk(df: pd.DataFrame, out_dir=None):
    if "merchant_risk_score" not in df.columns or "merchant_category" not in df.columns:
        print("Brak merchant_risk_score – pomijam plot.")
        return

    if out_dir is None:
        from .config import FIGURES_DIR
        out_dir = FIGURES_DIR

    risk_df = (
        df.groupby("merchant_category")["merchant_risk_score"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8,5))
    risk_df.plot(kind="bar", color="tomato")
    plt.title("Merchant Risk Score (Fraud Normalized 0-1)")
    plt.ylabel("Risk score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = out_dir / "merchant_risk_score.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved: {out}")


def plot_merchant_risk(df: pd.DataFrame) -> None:
    """
    Ranking kategorii merchant_category po fraud_rate (0-1).
    To jest "merchant_risk_score" w sensie analitycznym.
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


if __name__ == "__main__":
    main()
