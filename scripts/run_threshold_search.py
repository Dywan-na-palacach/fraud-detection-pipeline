from src.modeling import train_rf_model
from src.metrics_utils import best_threshold_by_cost, show_metrics


def main():
    model, X_te, y_te, proba, feats = train_rf_model()

    thr, cost = best_threshold_by_cost(y_te, proba, cost_fp=1.0, cost_fn=10.0)
    print(f"\nBest threshold (by simple cost): {thr:.3f}, cost={cost:.2f}")
    show_metrics(y_te, proba, thr=thr)


if __name__ == "__main__":
    main()
