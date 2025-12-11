import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

from .config import FIGURES_DIR

def show_metrics(y_true, y_proba, thr: float = 0.5) -> None:
    y_pred = (y_proba >= thr).astype(int)

    print(f"\n=== metrics @ threshold={thr:.2f} ===")
    print("confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nreport:")
    print(classification_report(y_true, y_pred, digits=4))

    roc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    print(f"ROC AUC: {roc:.4f}")
    print(f"PR AUC:  {ap:.4f}")


def best_threshold_by_cost(y_true, y_proba, cost_fp=1.0, cost_fn=10.0):
    """
    Bardzo prosty grid search po thresholdzie, z kosztem FP/FN.
    """
    from sklearn.metrics import confusion_matrix

    thrs = np.linspace(0.01, 0.99, 99)
    best_thr = None
    best_cost = None

    for t in thrs:
        y_hat = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        cost = cost_fp * fp + cost_fn * fn

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_thr = t

    return best_thr, best_cost

def compute_ks(y_true, y_proba):
    """
    Liczy statystykę KS (Kolmogorov–Smirnov) dla modelu scoringowego.
    Klasy:
      - y=0: "good" (non-fraud)
      - y=1: "bad" (fraud)

    Zwraca:
      ks_value (float), ks_at (score przy którym KS jest maksymalny)
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    # sortujemy wg score malejąco (typowo w scoringu)
    order = np.argsort(-y_proba)
    y_true_sorted = y_true[order]

    # liczba good/bad
    n_good = (y_true_sorted == 0).sum()
    n_bad = (y_true_sorted == 1).sum()

    if n_good == 0 or n_bad == 0:
        # w skrajnych przypadkach nie ma sensu liczyć KS
        return 0.0, None

    # kumulacyjne liczby good/bad
    cum_good = np.cumsum(y_true_sorted == 0) / n_good
    cum_bad = np.cumsum(y_true_sorted == 1) / n_bad

    # różnica dystrybuant
    diff = np.abs(cum_bad - cum_good)
    ks_value = diff.max()
    ks_idx = diff.argmax()
    ks_at = y_proba[order][ks_idx]

    return float(ks_value), float(ks_at)


def plot_ks(y_true, y_proba, model_name: str = "Model"):
    """
    Rysuje wykres KS (CDF good/bad + punkt maksymalnej różnicy)
    i zapisuje do reports/figures/ks_<model_name>.png
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    order = np.argsort(-y_proba)
    y_true_sorted = y_true[order]
    scores_sorted = y_proba[order]

    n_good = (y_true_sorted == 0).sum()
    n_bad = (y_true_sorted == 1).sum()

    if n_good == 0 or n_bad == 0:
        print(f"Nie można policzyć KS dla {model_name} – brak good/bad.")
        return

    cum_good = np.cumsum(y_true_sorted == 0) / n_good
    cum_bad = np.cumsum(y_true_sorted == 1) / n_bad
    diff = np.abs(cum_bad - cum_good)

    ks_value = diff.max()
    ks_idx = diff.argmax()
    ks_score = scores_sorted[ks_idx]

    # wykres
    plt.figure(figsize=(8, 5))
    plt.plot(cum_good, label="CDF good (non-fraud)")
    plt.plot(cum_bad, label="CDF bad (fraud)")
    plt.plot(
        ks_idx,
        cum_good[ks_idx],
        "o",
        label=f"KS={ks_value:.3f} @ score≈{ks_score:.3f}",
    )
    plt.vlines(ks_idx, cum_good[ks_idx], cum_bad[ks_idx], linestyles="dashed")
    plt.title(f"KS curve – {model_name}")
    plt.xlabel("Posortowane obserwacje (malejąco po score)")
    plt.ylabel("Cumulative proportion")
    plt.legend()
    plt.grid(alpha=0.3)
    out_path = FIGURES_DIR / f"ks_{model_name}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")

    return float(ks_value), float(ks_score)


def plot_roc_pr_curves(y_true, y_proba, model_name: str, out_dir=None) -> None:
    """
    Rysuje i zapisuje wykresy ROC oraz Precision-Recall dla danego modelu.
    """
    from sklearn.metrics import roc_curve, precision_recall_curve

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if out_dir is None:
        try:
            from .config import FIGURES_DIR
            out_dir = FIGURES_DIR
        except Exception:
            out_dir = None

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=model_name)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve - {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    if out_dir is not None:
        out_path = out_dir / f"roc_{model_name}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        print(f"Saved: {out_path}")
    plt.close()

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=model_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall curve - {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    if out_dir is not None:
        out_path = out_dir / f"pr_{model_name}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        print(f"Saved: {out_path}")
    plt.close()


def cost_sweep_report(
    y_true,
    y_proba,
    model_name: str = "Model",
    cost_fp: float = 1.0,
    cost_fn_grid=(5, 10, 20, 50, 100),
):
    """
    Sweep po różnych wartościach cost_fn przy stałym cost_fp.
    Wypisuje tabelę i zapisuje wykres recall vs cost_fn.

    cost_fn_grid – np. (5, 10, 20, 50, 100)
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    # żeby móc zapisać wykres
    try:
        from .config import FIGURES_DIR
        out_dir = FIGURES_DIR
    except Exception:
        out_dir = None

    print(f"\n=== Cost sweep for {model_name} (FP={cost_fp}) ===")
    print("cost_fn\tthr\tcost\trecall\tprecision\tFP\tFN")

    from sklearn.metrics import confusion_matrix

    fn_list = []
    thr_list = []
    cost_list = []
    rec_list = []
    prec_list = []

    for cfn in cost_fn_grid:
        thr, total_cost = best_threshold_by_cost(
            y_true, y_proba, cost_fp=cost_fp, cost_fn=cfn
        )
        y_hat = (y_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        print(
            f"{cfn:.1f}\t{thr:.3f}\t{total_cost:.1f}\t"
            f"{rec:.3f}\t{prec:.3f}\t{fp}\t{fn}"
        )

        fn_list.append(cfn)
        thr_list.append(thr)
        cost_list.append(total_cost)
        rec_list.append(rec)
        prec_list.append(prec)

    # prosty wykres: recall vs cost_fn (przy optymalnym thresholdzie)
    if out_dir is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(fn_list, rec_list, marker="o", label="Recall (fraud)")
        plt.xlabel("Cost_FN (relative to FP=1)")
        plt.ylabel("Recall at optimal threshold")
        plt.title(f"Cost sweep – {model_name}")
        plt.grid(alpha=0.3)
        plt.legend()
        out_path = out_dir / f"cost_sweep_{model_name}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved: {out_path}")
