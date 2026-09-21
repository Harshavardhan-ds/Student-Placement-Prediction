"""Features, model pipeline and helpers for the campus placement classifier.

Everything that touches preprocessing lives inside a scikit-learn ``Pipeline``.
That keeps scaling and encoding *inside* each cross-validation fold, so no
information from held-out rows leaks into training.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Placement_Data_Full_Class.csv"
MODEL_PATH = ROOT / "models" / "placement_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

RANDOM_STATE = 42
CV_FOLDS = 5
CV_REPEATS = 10

TARGET = "status"
POSITIVE_LABEL = "Placed"

# Columns that must never be model inputs:
#   sl_no  - a row id
#   salary - only exists for placed students, so it leaks the answer
# ``gender`` is also left out on purpose: dropping it changes cross-validated
# accuracy by less than the noise (see the notebook), and a tool that scores
# individual students should not use a protected attribute.
# To experiment with it, add "gender" to CATEGORICAL_FEATURES and retrain.
NUMERIC_FEATURES = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
CATEGORICAL_FEATURES = ["ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_LABELS = {
    "ssc_p": "Secondary school % (10th grade)",
    "ssc_b": "Secondary school board",
    "hsc_p": "Higher secondary % (12th grade)",
    "hsc_b": "Higher secondary board",
    "hsc_s": "Higher secondary stream",
    "degree_p": "Undergraduate degree %",
    "degree_t": "Undergraduate degree field",
    "workex": "Work experience",
    "etest_p": "Employability test %",
    "specialisation": "MBA specialisation",
    "mba_p": "MBA %",
    "gender": "Gender",
}


def candidate_models() -> dict:
    """Candidate classifiers, ordered roughly from simplest to most complex.

    The order matters: ``train.choose_model`` prefers the simplest model whose
    score is statistically indistinguishable from the best one.
    """
    return {
        "Logistic regression": LogisticRegression(max_iter=1000),
        "Gaussian naive Bayes": GaussianNB(),
        "Linear SVM": SVC(kernel="linear"),
        "RBF SVM": SVC(kernel="rbf"),
        "Random forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    }


def with_probabilities(model):
    """Return ``model``, wrapped so it has ``predict_proba`` if it lacks one.

    SVMs only produce a decision score. Scoring them in cross-validation does not
    need probabilities, but the app shows "chance of placement", so a deployed
    SVM is calibrated. (``SVC(probability=True)`` is deprecated as of scikit-learn 1.9.)
    """
    if hasattr(model, "predict_proba"):
        return model
    return CalibratedClassifierCV(model, ensemble=False)


def build_pipeline(model, numeric=None, categorical=None) -> Pipeline:
    """Preprocessing (scale numeric, one-hot categorical) followed by ``model``."""
    numeric = NUMERIC_FEATURES if numeric is None else numeric
    categorical = CATEGORICAL_FEATURES if categorical is None else categorical
    prep = ColumnTransformer(
        [
            ("num", StandardScaler(), list(numeric)),
            (
                "cat",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
                list(categorical),
            ),
        ]
    )
    return Pipeline([("prep", prep), ("model", model)])


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the CSV exactly as shipped (including ``salary`` and ``sl_no``)."""
    return pd.read_csv(path)


def load_data(path: Path | str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Return model inputs ``X`` and a 0/1 target ``y`` (1 = placed)."""
    df = load_raw(path)
    missing = [c for c in [*FEATURES, TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    X = df[FEATURES].copy()
    y = (df[TARGET] == POSITIVE_LABEL).astype(int)
    return X, y


def feature_ranges(X: pd.DataFrame) -> dict[str, list[float]]:
    """Min/max of each numeric feature, used to warn about out-of-range inputs."""
    return {c: [float(X[c].min()), float(X[c].max())] for c in NUMERIC_FEATURES}


def _readable(encoded_name: str) -> str:
    """Turn a ColumnTransformer output name into text a person can read."""
    if encoded_name.startswith("num__"):
        return FEATURE_LABELS[encoded_name[len("num__"):]]
    body = encoded_name[len("cat__"):]
    for col in CATEGORICAL_FEATURES:
        if body.startswith(col + "_"):
            return f"{FEATURE_LABELS[col]}: {body[len(col) + 1:]}"
    return body


def explain_prediction(pipeline: Pipeline, row: pd.DataFrame, reference: pd.DataFrame):
    """Per-feature push on the log-odds of placement, versus an average student.

    Only defined for linear models (logistic regression, linear SVM); returns
    ``None`` for anything else. Contributions are ``coef * (x - mean_x)``, where
    ``mean_x`` is the mean of the encoded reference data, so they sum exactly to
    ``decision_function(row) - mean(decision_function(reference))``.
    """
    model = pipeline.named_steps["model"]
    coef = getattr(model, "coef_", None)
    if coef is None or np.ndim(coef) != 2 or coef.shape[0] != 1:
        return None
    coef = np.asarray(coef)[0]
    prep = pipeline.named_steps["prep"]
    x = prep.transform(row)[0]
    baseline = prep.transform(reference).mean(axis=0)
    names = [_readable(n) for n in prep.get_feature_names_out()]
    out = pd.DataFrame({"feature": names, "contribution": coef * (x - baseline)})
    return out.reindex(out["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
