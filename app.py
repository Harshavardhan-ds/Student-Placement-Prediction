"""Streamlit app: estimate a student's chance of campus placement.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from placement.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURE_LABELS,
    NUMERIC_FEATURES,
    TARGET,
    explain_prediction,
    load_data,
    load_raw,
)
from placement.train import load_model

st.set_page_config(page_title="Student placement predictor", page_icon="🎓", layout="wide")

RAISES = "#2b6cb0"  # blue: pushes the chance up
LOWERS = "#dd8452"  # orange: pushes the chance down


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model()


@st.cache_data
def get_data():
    X, y = load_data()
    return X, y, load_raw()


pipeline, metrics = get_model()
X, y, raw = get_data()
ranges = (metrics or {}).get("feature_ranges", {})


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Student placement predictor")
st.write(
    "Estimate a student's chance of being placed through campus recruitment, "
    "based on their academic record."
)
st.info(
    "This is a learning project trained on 215 students from a single campus. "
    "Do not use it for admissions, hiring, or any decision about a real person.",
    icon="ℹ️",
)

tab_predict, tab_perf, tab_data = st.tabs(["Predict", "How well does it work?", "Explore the data"])

# --------------------------------------------------------------------------- #
# Predict
# --------------------------------------------------------------------------- #
with tab_predict:
    form_col, result_col = st.columns([1, 1], gap="large")

    with form_col:
        with st.form("student"):
            st.subheader("Student record")

            def pct(col: str) -> float:
                """Percentage slider that starts at the dataset median."""
                default = round(float(X[col].median()) * 2) / 2
                return st.slider(FEATURE_LABELS[col], 0.0, 100.0, default, step=0.5)

            c1, c2 = st.columns(2)
            with c1:
                ssc_p = pct("ssc_p")
                hsc_p = pct("hsc_p")
                degree_p = pct("degree_p")
                etest_p = pct("etest_p")
                mba_p = pct("mba_p")
            with c2:
                choices = {c: sorted(X[c].unique().tolist()) for c in CATEGORICAL_FEATURES}
                ssc_b = st.selectbox(FEATURE_LABELS["ssc_b"], choices["ssc_b"])
                hsc_b = st.selectbox(FEATURE_LABELS["hsc_b"], choices["hsc_b"])
                hsc_s = st.selectbox(FEATURE_LABELS["hsc_s"], choices["hsc_s"])
                degree_t = st.selectbox(FEATURE_LABELS["degree_t"], choices["degree_t"])
                specialisation = st.selectbox(FEATURE_LABELS["specialisation"], choices["specialisation"])
                workex = st.selectbox(FEATURE_LABELS["workex"], choices["workex"])

            submitted = st.form_submit_button("Predict placement", type="primary")

    with result_col:
        st.subheader("Result")
        if not submitted:
            st.write("Fill in the student's record and choose **Predict placement**.")
        else:
            row = pd.DataFrame(
                [
                    {
                        "ssc_p": ssc_p,
                        "hsc_p": hsc_p,
                        "degree_p": degree_p,
                        "etest_p": etest_p,
                        "mba_p": mba_p,
                        "ssc_b": ssc_b,
                        "hsc_b": hsc_b,
                        "hsc_s": hsc_s,
                        "degree_t": degree_t,
                        "workex": workex,
                        "specialisation": specialisation,
                    }
                ]
            )[[*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]

            p_placed = float(pipeline.predict_proba(row)[0, 1])
            # Never show 0% or 100%: a model fit on 215 rows is not that certain.
            shown = ">99%" if p_placed > 0.99 else "<1%" if p_placed < 0.01 else f"{p_placed:.0%}"
            st.metric("Estimated chance of placement", shown)
            st.progress(p_placed)
            if p_placed >= 0.5:
                st.success("The model predicts this student would be placed.")
            else:
                st.warning("The model predicts this student would not be placed.")

            outside = [
                f"{FEATURE_LABELS[c]} ({row.at[0, c]:g}; training data spans {lo:g} to {hi:g})"
                for c, (lo, hi) in ranges.items()
                if not lo <= row.at[0, c] <= hi
            ]
            if outside:
                st.warning(
                    "Some values are outside the range the model was trained on, so the "
                    "estimate is less reliable: " + "; ".join(outside) + "."
                )

            contrib = explain_prediction(pipeline, row, X)
            if contrib is not None:
                st.markdown("**What moved the estimate**")
                st.caption(
                    "Effect of each input on the odds of placement, compared with the average student "
                    "in the dataset. These are patterns in a small dataset, not cause and effect. "
                    "For example, MBA % points the opposite way from what you might expect once school "
                    "and degree marks are accounted for, so do not read the chart as advice."
                )
                top = contrib.head(8).copy()
                top["direction"] = top["contribution"].map(lambda v: "Raises chance" if v > 0 else "Lowers chance")
                chart = (
                    alt.Chart(top)
                    .mark_bar()
                    .encode(
                        x=alt.X("contribution:Q", title="Effect on log-odds of placement"),
                        y=alt.Y("feature:N", sort=None, title=None),
                        color=alt.Color(
                            "direction:N",
                            scale=alt.Scale(domain=["Raises chance", "Lowers chance"], range=[RAISES, LOWERS]),
                            legend=alt.Legend(title=None, orient="bottom"),
                        ),
                        tooltip=["feature", alt.Tooltip("contribution:Q", format="+.2f")],
                    )
                    .properties(height=260, width="container")
                )
                st.altair_chart(chart)

# --------------------------------------------------------------------------- #
# Model performance
# --------------------------------------------------------------------------- #
with tab_perf:
    if not metrics:
        st.warning("No saved metrics found. Run `python -m placement.train` to generate them.")
    else:
        chosen = metrics["chosen_model"]
        cand = pd.DataFrame(metrics["candidates"])
        row_c = cand[cand["model"] == chosen].iloc[0]
        cv = metrics["cv"]

        st.subheader(f"Deployed model: {chosen}")
        st.write(
            f"Scores are averages over {cv['repeats']} repeats of {cv['folds']}-fold cross-validation, "
            "so every student is always scored by a model that never saw them."
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{row_c['accuracy_mean']:.1%}", f"± {row_c['accuracy_std']:.1%}", delta_color="off")
        m2.metric("Precision (placed)", f"{row_c['precision_mean']:.1%}")
        m3.metric("Recall (placed)", f"{row_c['recall_mean']:.1%}")
        m4.metric("ROC AUC", f"{row_c['roc_auc_mean']:.3f}")
        st.caption(
            f"Always guessing \"placed\" would be right {metrics['majority_baseline']:.1%} of the time, "
            "so that is the score to beat."
        )

        st.subheader("Model comparison")
        bars = (
            alt.Chart(cand)
            .mark_bar(color=RAISES)
            .encode(
                x=alt.X("accuracy_mean:Q", title="Cross-validated accuracy", scale=alt.Scale(domain=[0.5, 1])),
                y=alt.Y("model:N", sort=None, title=None),
                tooltip=["model", alt.Tooltip("accuracy_mean:Q", format=".3f"), alt.Tooltip("accuracy_std:Q", format=".3f")],
            )
        )
        base_rule = (
            alt.Chart(pd.DataFrame({"x": [metrics["majority_baseline"]]}))
            .mark_rule(color=LOWERS, strokeDash=[6, 4])
            .encode(x="x:Q")
        )
        st.altair_chart((bars + base_rule).properties(height=220, width="container"))
        st.caption("Dashed line: the always-guess-placed baseline.")

        table = pd.DataFrame(
            {
                "Model": cand["model"],
                "Accuracy": [f"{m:.3f} ± {s:.3f}" for m, s in zip(cand["accuracy_mean"], cand["accuracy_std"])],
                "F1": [f"{m:.3f} ± {s:.3f}" for m, s in zip(cand["f1_mean"], cand["f1_std"])],
                "ROC AUC": [f"{m:.3f} ± {s:.3f}" for m, s in zip(cand["roc_auc_mean"], cand["roc_auc_std"])],
            }
        )
        st.dataframe(table, hide_index=True)
        st.markdown(
            f"**Why {chosen}?** {metrics['selection_rule']} The top models differ by far less than the "
            "spread between folds, so the simpler, more interpretable one wins. Only naive Bayes is "
            "clearly behind."
        )

        st.subheader("Where the deployed model gets it wrong")
        cm = metrics["out_of_fold"]["confusion_matrix"]
        labels = metrics["out_of_fold"]["labels"]
        st.dataframe(
            pd.DataFrame(
                cm,
                index=[f"Actually {l.lower()}" for l in labels],
                columns=[f"Predicted {l.lower()}" for l in labels],
            )
        )
        st.caption(
            "Counts of all 215 students, each scored by a model trained without them. "
            "With this little data, treat any score as accurate to roughly ±5 percentage points."
        )

# --------------------------------------------------------------------------- #
# Explore the data
# --------------------------------------------------------------------------- #
with tab_data:
    st.subheader("Placement by feature")
    options = {FEATURE_LABELS[c]: c for c in [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]}
    label = st.selectbox("Feature", list(options))
    col = options[label]
    plot_df = raw[[col, TARGET]].copy()

    if col in NUMERIC_FEATURES:
        hist = (
            alt.Chart(plot_df)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=20), title=label),
                y=alt.Y("count()", stack=None, title="Students"),
                color=alt.Color(f"{TARGET}:N", title=None, scale=alt.Scale(domain=["Placed", "Not Placed"], range=[RAISES, LOWERS])),
            )
            .properties(height=300, width="container")
        )
        st.altair_chart(hist)
    else:
        rate = plot_df.groupby(col)[TARGET].agg(
            placed=lambda s: (s == "Placed").mean(), students="size"
        ).reset_index()
        bars = (
            alt.Chart(rate)
            .mark_bar(color=RAISES)
            .encode(
                x=alt.X(f"{col}:N", title=label, sort="-y"),
                y=alt.Y("placed:Q", title="Share placed", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                tooltip=[col, alt.Tooltip("placed:Q", format=".0%"), "students"],
            )
            .properties(height=300, width="container")
        )
        st.altair_chart(bars)
        st.caption("Small groups give noisy percentages, so check the student counts in the tooltip.")

    with st.expander("Show the raw dataset"):
        st.dataframe(raw, hide_index=True)
        st.caption(
            "Source: Factors Affecting Campus Placement (Kaggle, by Ben Roshan). "
            "Salary is empty for students who were not placed, which is why it is not used as an input."
        )
