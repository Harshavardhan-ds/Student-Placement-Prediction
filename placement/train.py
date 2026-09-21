"""Compare candidate models, pick one, and save it.

Run from the repository root:

    python -m placement.train

Method
------
* Every candidate is a full pipeline (scaling + encoding + classifier), scored
  with repeated stratified 5-fold cross-validation on all rows. With only 215
  students a single 80/20 split leaves ~43 test rows, where one student is
  worth 2+ accuracy points, so a single split cannot separate these models.
* Among models whose mean accuracy is within one standard error of the best,
  the simplest one is chosen (the "one-standard-error rule").
* The chosen model's confusion matrix comes from out-of-fold predictions, so
  every student is scored by a model that never saw them.
* The deployed model is then refit on all rows.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from .pipeline import (
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    CV_REPEATS,
    DATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    build_pipeline,
    candidate_models,
    feature_ranges,
    load_data,
    with_probabilities,
)

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}
SELECTION_RULE = (
    "Simplest model whose mean CV accuracy is within one standard error "
    "(std / sqrt(folds)) of the best model."
)
DEFAULT_MODEL = "Logistic regression"


def choose_model(results: list[dict], n_splits: int = CV_FOLDS) -> str:
    """One-standard-error rule. ``results`` must be ordered simplest first."""
    best = max(results, key=lambda r: r["accuracy_mean"])
    threshold = best["accuracy_mean"] - best["accuracy_std"] / math.sqrt(n_splits)
    for r in results:
        if r["accuracy_mean"] >= threshold:
            return r["model"]
    return best["model"]  # unreachable, kept for safety


def fit_final(model_name: str = DEFAULT_MODEL, data_path: Path | str = DATA_PATH):
    """Fit the named candidate on every row and return the fitted pipeline."""
    X, y = load_data(data_path)
    return build_pipeline(with_probabilities(candidate_models()[model_name])).fit(X, y)


def train(
    data_path: Path | str = DATA_PATH,
    model_path: Path | str = MODEL_PATH,
    metrics_path: Path | str = METRICS_PATH,
    verbose: bool = True,
) -> dict:
    """Run the comparison, save the fitted model and a metrics JSON, return metrics."""
    X, y = load_data(data_path)
    cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)

    results = []
    for name, model in candidate_models().items():
        scores = cross_validate(build_pipeline(model), X, y, cv=cv, scoring=SCORING)
        row = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = round(float(np.mean(values)), 4)
            row[f"{metric}_std"] = round(float(np.std(values)), 4)
        results.append(row)
        if verbose:
            print(f"{name:22s} accuracy {row['accuracy_mean']:.3f} ± {row['accuracy_std']:.3f}")

    chosen = choose_model(results)

    # Out-of-fold predictions for the chosen model: each row is scored by a
    # model that was trained without it.
    oof_proba = cross_val_predict(
        build_pipeline(with_probabilities(candidate_models()[chosen])),
        X,
        y,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        method="predict_proba",
    )[:, 1]
    oof_pred = (oof_proba >= 0.5).astype(int)
    report = classification_report(
        y, oof_pred, target_names=["Not placed", "Placed"], output_dict=True, zero_division=0
    )

    pipeline = fit_final(chosen, data_path)
    metrics = {
        "sklearn_version": sklearn.__version__,
        "n_rows": int(len(y)),
        "n_placed": int(y.sum()),
        "majority_baseline": round(float(max(y.mean(), 1 - y.mean())), 4),
        "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
        "feature_ranges": feature_ranges(X),
        "cv": {"folds": CV_FOLDS, "repeats": CV_REPEATS},
        "candidates": results,
        "chosen_model": chosen,
        "selection_rule": SELECTION_RULE,
        "out_of_fold": {
            "labels": ["Not placed", "Placed"],
            "confusion_matrix": confusion_matrix(y, oof_pred).tolist(),
            "report": {
                k: {m: round(float(v), 4) for m, v in vals.items()}
                for k, vals in report.items()
                if isinstance(vals, dict)
            },
            "accuracy": round(float(report["accuracy"]), 4),
        },
    }

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    Path(metrics_path).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if verbose:
        print(f"\nChosen model: {chosen}  ({SELECTION_RULE})")
        print(f"Saved {model_path} and {metrics_path}")
    return metrics


def load_model(
    model_path: Path | str = MODEL_PATH, metrics_path: Path | str = METRICS_PATH
):
    """Load the saved pipeline and metrics, refitting quickly if needed.

    A pickled scikit-learn model is only guaranteed to load under the version
    that wrote it. If the file is missing, unreadable, or was written by a
    different scikit-learn, the chosen model is refit from the CSV (about a
    second) instead of failing.
    """
    metrics = None
    if Path(metrics_path).exists():
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))

    same_version = metrics is not None and metrics.get("sklearn_version") == sklearn.__version__
    if same_version and Path(model_path).exists():
        try:
            return joblib.load(model_path), metrics
        except Exception:  # noqa: BLE001 - any unpickling failure means "refit"
            pass

    name = metrics["chosen_model"] if metrics else DEFAULT_MODEL
    return fit_final(name), metrics


if __name__ == "__main__":
    train()
