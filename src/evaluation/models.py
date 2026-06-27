"""Model factories used by downstream harmonization tasks."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


def _model(seed: int = RANDOM_SEED, kind: str = "rf") -> Pipeline:
    """Classification model factory.

    kind:
      - "rf"   : RandomForestClassifier (legacy default)
      - "lgbm" : LightGBM sensitivity check
      - "mlp"  : sklearn MLPClassifier for pretrain-finetune protocols
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    if kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    elif kind == "lgbm":
        from lightgbm import LGBMClassifier

        clf = LGBMClassifier(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.05,
            min_child_samples=5,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
    elif kind == "mlp":
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=400,
            early_stopping=False,
            warm_start=True,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown classification model kind: {kind}")
    return Pipeline([("imputer", imputer), ("scaler", scaler), ("clf", clf)])


def _regression_model(seed: int = RANDOM_SEED, kind: str = "rf") -> Pipeline:
    """Regression model factory. See `_model` for the kind contract."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    if kind == "rf":
        reg = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=1,
        )
    elif kind == "lgbm":
        from lightgbm import LGBMRegressor

        reg = LGBMRegressor(
            n_estimators=400,
            num_leaves=31,
            learning_rate=0.05,
            min_child_samples=3,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
    elif kind == "mlp":
        from sklearn.neural_network import MLPRegressor

        reg = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=400,
            early_stopping=False,
            warm_start=True,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown regression model kind: {kind}")
    return Pipeline([("imputer", imputer), ("scaler", scaler), ("reg", reg)])
