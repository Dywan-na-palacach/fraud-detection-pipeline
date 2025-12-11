from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_logreg_model():
    """
    Logistic Regression baseline with scaling.
    Działa na X z build_X_y (same features co RandomForest).
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            max_iter=3000,   # trochę więcej iteracji
            C=3.0,           # słabsza regularyzacja → model może wyciągnąć trochę więcej sygnału
        )),
    ])
    return model


def build_rf_model(seed: int = 42):
    """
    RandomForest na tych samych cechach co LogReg (X z build_X_y).
    Lekko ograniczamy złożoność (min_samples_leaf), żeby zmniejszyć ryzyko przeuczenia
    przy bardzo nierównych klasach.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=5,   # nowe: liść musi mieć min. 5 obserwacji
        # max_depth=None zostawiamy jako domyślne
    )
    return model
