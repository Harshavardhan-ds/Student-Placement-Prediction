# Student placement prediction

Predict whether a student will be placed through campus recruitment from their academic record, work experience and MBA specialisation. Includes a Streamlit app, a training script, and an analysis notebook.

> **Learning project.** The model is trained on 215 students from one campus. Do not use it for admissions, hiring, or any decision about a real person.

## What's inside

| | |
|---|---|
| `app.py` | Streamlit app: enter a student's record, get an estimated chance of placement and see what drove it |
| `student_placement_prediction.ipynb` | Data exploration, model comparison and interpretation |
| `placement/` | Feature definitions, the model pipeline, and the training script |
| `models/` | The trained model and its metrics, so the app starts instantly |
| `tests/` | Automated tests |

## Results

Five classifiers were compared with 5-fold cross-validation repeated 10 times on all 215 students. Scaling and encoding happen inside each fold, so nothing leaks from the held-out data.

| Model | Accuracy (%) | F1 | ROC AUC |
|---|---|---|---|
| Logistic regression (deployed) | 86.6 ± 4.9 | 0.905 | 0.938 |
| Gaussian naive Bayes | 80.5 ± 5.0 | 0.860 | 0.861 |
| Linear SVM | 86.9 ± 4.3 | 0.906 | 0.935 |
| RBF SVM | 84.2 ± 4.6 | 0.891 | 0.917 |
| Random forest | 85.2 ± 4.7 | 0.897 | 0.918 |

Always guessing "Placed" would be right 68.8% of the time, so the models add real signal.

Logistic regression, linear SVM, RBF SVM and random forest are **statistically tied**: the gaps between them are smaller than the fold-to-fold spread. The deployed model is the simplest one within one standard error of the best, which also gives calibrated probabilities and readable coefficients. Only naive Bayes is clearly worse.

## Quick start

```bash
git clone https://github.com/<your-username>/Student-Placement-Prediction.git
cd Student-Placement-Prediction

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Retrain the model

```bash
python -m placement.train
```

This re-runs the model comparison, picks a model, and rewrites `models/placement_model.joblib` and `models/metrics.json`, which the app reads.

A saved scikit-learn model only loads reliably under the version that wrote it. If yours differs, the app refits the chosen model from the CSV on startup (about a second) instead of failing.

## Run the notebook

```bash
pip install -r requirements-dev.txt
jupyter notebook student_placement_prediction.ipynb
```

Start Jupyter from the repository root so that `import placement` works.

## Run the tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Deploy the app for free

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), choose **New app**, and select your repository.
3. Set the main file path to `app.py` and deploy.

No secrets or extra configuration are needed.

## Design decisions

- **`salary` is not a feature.** It is filled in for every placed student and for no unplaced one, so it would leak the answer.
- **`gender` is not a feature.** Adding it changes cross-validated accuracy by less than one point, well inside the noise, and a tool that scores individual students should not use a protected attribute for no measurable gain. To experiment, add `"gender"` to `CATEGORICAL_FEATURES` in `placement/pipeline.py` and retrain.
- **No rows are removed as outliers.** High marks are legitimate values, and with 215 rows every student counts.
- **Cross-validation instead of one split.** With about 43 test students, one student is worth more than two accuracy points. Repeated cross-validation is a far steadier yardstick.

## Limitations

- Treat any accuracy figure as good to roughly **±5 percentage points**.
- One campus, one dataset. The model has not been tested on other institutions, years or programmes.
- Some groups are tiny (12th-grade Arts: 11 students; degree field "Others": 11 students), so the model barely constrains them.
- The model finds correlation, not cause. For example, once school and degree marks are accounted for, a higher MBA % is associated with *slightly lower* placement in this data. That pattern is stable across bootstrap resamples, but it is a quirk of these 215 students, not advice.

## Data

[Factors Affecting Campus Placement](https://www.kaggle.com/benroshan/factors-affecting-campus-placement) on Kaggle, by Ben Roshan. The dataset belongs to its original authors; check the terms on its Kaggle page before redistributing it. The MIT license in this repository covers the code only.

## License

Code released under the [MIT License](LICENSE).
