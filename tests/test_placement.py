import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from placement.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    build_pipeline,
    explain_prediction,
    load_data,
    load_raw,
)
from placement.train import choose_model, fit_final, load_model, train


APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(scope="module")
def data():
    return load_data()


# --------------------------------------------------------------------------- #
# Data and features
# --------------------------------------------------------------------------- #
def test_dataset_shape_and_target(data):
    X, y = data
    assert X.shape == (215, len(FEATURES))
    assert set(y.unique()) == {0, 1}
    assert int(y.sum()) == 148  # placed students in the source CSV
    assert not X.isna().any().any()


def test_leaky_or_sensitive_columns_are_not_inputs(data):
    X, _ = data
    for col in ("salary", "sl_no", "status", "gender"):
        assert col not in X.columns


def test_salary_is_only_missing_for_unplaced_students():
    """Why salary must not be a feature: its presence reveals the label."""
    raw = load_raw()
    assert raw.loc[raw["status"] == "Not Placed", "salary"].isna().all()
    assert raw.loc[raw["status"] == "Placed", "salary"].notna().all()


# --------------------------------------------------------------------------- #
# Pipeline correctness
# --------------------------------------------------------------------------- #
def test_scaler_is_fit_on_training_rows_only(data):
    """Regression test for the original notebook, which scaled before splitting."""
    X, y = data
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    scaler = pipe.named_steps["prep"].named_transformers_["num"]
    np.testing.assert_allclose(scaler.mean_, X_train[NUMERIC_FEATURES].mean().to_numpy())
    assert not np.allclose(scaler.mean_, X[NUMERIC_FEATURES].mean().to_numpy())


def test_unseen_category_does_not_crash(data):
    X, y = data
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    row = X.iloc[[0]].copy()
    row["hsc_s"] = "Never seen before"
    assert 0.0 <= pipe.predict_proba(row)[0, 1] <= 1.0


def test_explanations_sum_to_logit_difference(data):
    X, y = data
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    row = X.iloc[[3]]
    contrib = explain_prediction(pipe, row, X)
    expected = pipe.decision_function(row)[0] - pipe.decision_function(X).mean()
    assert contrib["contribution"].sum() == pytest.approx(expected)
    assert contrib["feature"].is_unique
    assert (contrib["contribution"].abs().diff().dropna() <= 1e-12).all()  # sorted by size


def test_explanations_unavailable_for_non_linear_models(data):
    from sklearn.ensemble import RandomForestClassifier

    X, y = data
    pipe = build_pipeline(RandomForestClassifier(n_estimators=10, random_state=0)).fit(X, y)
    assert explain_prediction(pipe, X.iloc[[0]], X) is None


# --------------------------------------------------------------------------- #
# Model selection
# --------------------------------------------------------------------------- #
def _r(name, mean, std=0.04):
    return {"model": name, "accuracy_mean": mean, "accuracy_std": std}


def test_choose_model_prefers_simplest_within_one_standard_error():
    results = [_r("simple", 0.870), _r("complex", 0.875)]  # 0.005 gap < 0.04/sqrt(5)
    assert choose_model(results) == "simple"


def test_choose_model_takes_clear_winner():
    results = [_r("simple", 0.80), _r("complex", 0.90)]
    assert choose_model(results) == "complex"


# --------------------------------------------------------------------------- #
# End-to-end training and loading
# --------------------------------------------------------------------------- #
def test_train_writes_artifacts_and_beats_baseline(tmp_path):
    model_path, metrics_path = tmp_path / "m.joblib", tmp_path / "metrics.json"
    metrics = train(model_path=model_path, metrics_path=metrics_path, verbose=False)

    assert model_path.exists() and metrics_path.exists()
    assert json.loads(metrics_path.read_text())["chosen_model"] == metrics["chosen_model"]

    chosen = next(c for c in metrics["candidates"] if c["model"] == metrics["chosen_model"])
    assert chosen["accuracy_mean"] > metrics["majority_baseline"] + 0.10
    assert sum(map(sum, metrics["out_of_fold"]["confusion_matrix"])) == 215

    model, loaded = load_model(model_path, metrics_path)
    X, _ = load_data()
    assert model.predict_proba(X.iloc[:5]).shape == (5, 2)
    assert loaded["chosen_model"] == metrics["chosen_model"]


def test_load_model_refits_when_artifact_is_missing_or_corrupt(tmp_path):
    missing, metrics_path = tmp_path / "nope.joblib", tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"chosen_model": "Logistic regression", "sklearn_version": "0.0"}))
    model, _ = load_model(missing, metrics_path)
    X, _ = load_data()
    assert model.predict_proba(X.iloc[:2]).shape == (2, 2)

    corrupt = tmp_path / "bad.joblib"
    corrupt.write_bytes(b"not a pickle")
    import sklearn

    metrics_path.write_text(
        json.dumps({"chosen_model": "Logistic regression", "sklearn_version": sklearn.__version__})
    )
    model, _ = load_model(corrupt, metrics_path)  # same version, but unreadable -> refit
    assert model.predict_proba(X.iloc[:2]).shape == (2, 2)


def test_committed_model_matches_committed_metrics():
    """The shipped artifacts must load and agree with each other."""
    model, metrics = load_model()
    assert metrics is not None
    assert type(model.named_steps["model"]).__name__ in {
        "LogisticRegression", "SVC", "GaussianNB", "RandomForestClassifier"
    }
    assert set(metrics["features"]["numeric"]) == set(NUMERIC_FEATURES)
    assert set(metrics["features"]["categorical"]) == set(CATEGORICAL_FEATURES)


def test_fit_final_predicts_sensibly():
    X, y = load_data()
    model = fit_final("Logistic regression")
    assert (model.predict(X) == y).mean() > 0.80


# --------------------------------------------------------------------------- #
# Streamlit app
# --------------------------------------------------------------------------- #
def test_streamlit_app_runs_and_predicts():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Predict", "How well does it work?", "Explore the data"]

    at.button[0].click().run()
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert "Estimated chance of placement" in labels
    assert labels["Estimated chance of placement"].strip("<>%").isdigit()


def test_streamlit_app_warns_about_out_of_range_input():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60).run()
    for s in at.slider:
        s.set_value(5.0)
    at.button[0].click().run()
    assert not at.exception
    assert any("outside the range" in w.value for w in at.warning)


# --------------------------------------------------------------------------- #
# A deployed SVM must still give probabilities (SVC(probability=True) is deprecated)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["Linear SVM", "RBF SVM"])
def test_svm_can_be_deployed_with_probabilities(name):
    X, _ = load_data()
    model = fit_final(name)
    proba = model.predict_proba(X.iloc[:5])
    assert proba.shape == (5, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert explain_prediction(model, X.iloc[[0]], X) is None  # calibrated wrapper has no coef_


def test_train_works_when_an_svm_is_chosen(tmp_path, monkeypatch):
    from sklearn.svm import SVC

    import placement.train as train_module

    monkeypatch.setattr(
        train_module,
        "candidate_models",
        lambda: {"Logistic regression": LogisticRegression(max_iter=1000), "Linear SVM": SVC(kernel="linear")},
    )
    monkeypatch.setattr(train_module, "choose_model", lambda results, n_splits=5: "Linear SVM")
    metrics = train_module.train(model_path=tmp_path / "m.joblib", metrics_path=tmp_path / "x.json", verbose=False)
    assert metrics["chosen_model"] == "Linear SVM"
    assert sum(map(sum, metrics["out_of_fold"]["confusion_matrix"])) == 215
