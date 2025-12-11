from __future__ import annotations

from sklearn.model_selection import train_test_split

from src.load_data import load_transactions
from src.features import build_X_y
from src.modeling import build_logreg_model, build_rf_model
from src.metrics_utils import (
    show_metrics,
    best_threshold_by_cost,
    compute_ks,
    plot_ks,
    plot_roc_pr_curves,
    cost_sweep_report,
)


def evaluate(model, X_train, y_train, X_test, y_test, name: str):
    print(f"\n=== {name} ===")

    # trenowanie
    model.fit(X_train, y_train)

    # predykcje prawdopodobieństw
    y_proba = model.predict_proba(X_test)[:, 1]

    # metryki @ 0.5
    show_metrics(y_test, y_proba, thr=0.5)

    # threshold kosztowy
    thr, cost = best_threshold_by_cost(y_test, y_proba, cost_fp=1.0, cost_fn=10.0)
    print(f"\nBest threshold (FP=1, FN=10): {thr:.2f}, cost={cost:.2f}")
    show_metrics(y_test, y_proba, thr=thr)

    # KS
    ks_value, ks_thr = compute_ks(y_test, y_proba)
    print(f"KS statistic: {ks_value:.4f} at threshold={ks_thr:.4f}")
    plot_ks(y_test, y_proba, name)

    # ROC + PR
    plot_roc_pr_curves(y_test, y_proba, model_name=name)

    # sweep po różnych kosztach FN, wykres(żeby zobaczyć, od kiedy model cos wylapie fraudy)
    cost_sweep_report(
        y_test,
        y_proba,
        model_name=name,
        cost_fp=1.0,
        cost_fn_grid=(5, 10, 20, 50, 100),
    )


def main():
    # 1. wczytanie danych
    df = load_transactions()
    print(f"Loaded df: {df.shape}")

    # 2. budowa cech
    X, y, feats = build_X_y(df)
    print(f"X: {X.shape}, features={len(feats)}, fraud_rate={y.mean():.4f}")

    # 3. podział train/test (WSPÓLNY dla obu modeli)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train={X_train.shape}, Test={X_test.shape}")

    # 4. Logistic Regression baseline (ze skalowaniem)
    logreg = build_logreg_model()
    evaluate(logreg, X_train, y_train, X_test, y_test, "LogisticRegression")

    # 5. Random Forest
    rf = build_rf_model()
    evaluate(rf, X_train, y_train, X_test, y_test, "RandomForest")




if __name__ == "__main__":
    main()
