import numpy as np
import pandas as pd
import pytest

from src.evaluation.downstream.correction import apply_train_only_correction


def test_single_batch_correction_is_skipped():
    train = pd.DataFrame({"source": ["DFL", "DFL"], "x": [1.0, 2.0]})
    test = pd.DataFrame({"source": ["DFL"], "x": [3.0]})

    result = apply_train_only_correction(train, test, ["x"], method="combat_p")

    assert result.applied is False
    assert result.reason == "single_train_batch"
    assert result.train_df.equals(train)
    assert result.test_df.equals(test)


def test_parametric_combat_preserves_shape_and_finite_values():
    train = pd.DataFrame(
        {
            "source": ["DFL"] * 5 + ["SkillCorner"] * 5,
            "x": [1, 2, 3, 4, 5, 10, 11, 12, 13, 14],
            "y": [0, 1, 0, 1, 0, 2, 3, 2, 3, 2],
        }
    )
    test = pd.DataFrame({"source": ["DFL", "SkillCorner"], "x": [3, 12], "y": [1, 3]})

    result = apply_train_only_correction(train, test, ["x", "y"], method="combat_p")

    assert result.applied is True
    assert result.reason == "applied"
    assert result.train_df.shape == train.shape
    assert result.test_df.shape == test.shape
    assert result.train_df[["x", "y"]].notna().all().all()
    assert result.test_df[["x", "y"]].notna().all().all()


def test_parametric_combat_matches_neurocombat_sklearn_reference():
    neurocombat = pytest.importorskip("neurocombat_sklearn")

    rng = np.random.default_rng(42)
    n_dfl, n_sc, n_features = 50, 80, 18
    feature_cols = [f"f{i}" for i in range(n_features)]

    train_x = np.vstack(
        [
            rng.normal(0.0, 1.0, (n_dfl, n_features)),
            rng.normal(2.0, 1.5, (n_sc, n_features)),
        ]
    )
    train = pd.DataFrame(train_x, columns=feature_cols)
    train["source"] = ["DFL"] * n_dfl + ["SkillCorner"] * n_sc

    test_x = np.vstack(
        [
            rng.normal(0.0, 1.0, (10, n_features)),
            rng.normal(2.0, 1.5, (15, n_features)),
        ]
    )
    test = pd.DataFrame(test_x, columns=feature_cols)
    test["source"] = ["DFL"] * 10 + ["SkillCorner"] * 15

    result = apply_train_only_correction(train, test, feature_cols, method="combat_p")

    train_sites = (
        (train["source"] == "SkillCorner").to_numpy().astype(int).reshape(-1, 1)
    )
    test_sites = (
        (test["source"] == "SkillCorner").to_numpy().astype(int).reshape(-1, 1)
    )
    model = neurocombat.CombatModel()
    try:
        model.fit(
            train[feature_cols].to_numpy(),
            sites=train_sites,
            discrete_covariates=None,
            continuous_covariates=None,
        )
    except TypeError as exc:
        pytest.skip(f"installed neurocombat-sklearn is incompatible: {exc}")
    reference_train = model.transform(
        train[feature_cols].to_numpy(),
        sites=train_sites,
        discrete_covariates=None,
        continuous_covariates=None,
    )
    reference_test = model.transform(
        test[feature_cols].to_numpy(),
        sites=test_sites,
        discrete_covariates=None,
        continuous_covariates=None,
    )

    np.testing.assert_allclose(
        result.train_df[feature_cols].to_numpy(), reference_train, atol=1e-6
    )
    np.testing.assert_allclose(
        result.test_df[feature_cols].to_numpy(), reference_test, atol=1e-6
    )
