from src.load_data import load_transactions
from src.features import (
    build_X_y,
    add_time_cols,
    add_numeric_stuff,
    add_amount_deviation_feature,
)
from src.eda import (
    basic_stats,
    plot_class_balance,
    plot_hourly_fraud_rate,
    fraud_rate_by_column,
    plot_fraud_vs_log_amount,
    plot_amount_hist_overlay,
    plot_fraud_vs_amount_deviation,
    plot_boxplots_for_features,
    plot_corr_heatmap,
    plot_pareto_merchant_category,
    plot_merchant_risk,
)



def main():
    # wczytanie danych (z DEBUG_SMALL_DATA lub pełnych - zależnie od config.py)
    df = load_transactions()

    # printy / podstawy
    basic_stats(df)
    plot_class_balance(df)
    plot_hourly_fraud_rate(df)

    # klasyczne pivoty po kategoriach
    for col in ["payment_channel", "merchant_category", "location", "device_used"]:
        if col in df.columns:
            fraud_rate_by_column(df, col, top_n=10)

    # budowa X, y -> po drodze sprawdzamy, że pipeline featurów działa
    X, y, feats = build_X_y(df)
    print(f"Built feature matrix X with shape={X.shape}, n_features={len(feats)}")

    # przygotowujemy df z tymi samymi feature'ami do EDA feature-behaviour
    df_feat = df.copy()
    df_feat = add_time_cols(df_feat)
    df_feat = add_numeric_stuff(df_feat)
    df_feat = add_amount_deviation_feature(df_feat)

    # EDA bardziej „data science” na df_feat:
    plot_fraud_vs_log_amount(df_feat)
    plot_amount_hist_overlay(df_feat)
    plot_fraud_vs_amount_deviation(df_feat)

    # Boxploty dla kilku ważnych cech
    box_cols = [
        "amount",
        "log_amount",
        "time_since_last_transaction",
        "spending_deviation_score",
        "velocity_score",
        "geo_anomaly_score",
        "amount_dev_robust",
    ]
    box_cols = [c for c in box_cols if c in df_feat.columns]
    if box_cols:
        plot_boxplots_for_features(df_feat, box_cols)

    # korelacje
    plot_corr_heatmap(df_feat)

    # Pareto dla merchant_category (może działać na surowym df)
    plot_pareto_merchant_category(df)
    plot_merchant_risk(df)



if __name__ == "__main__":
    main()
