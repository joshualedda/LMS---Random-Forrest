"""Compare four classifiers on the same held-out student test set."""

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
FEATURES = [
    "logins_per_week", "avg_session_min", "avg_quiz_score",
    "avg_assignment_score", "late_submission_rate", "missed_deadlines",
    "forum_posts", "days_since_last_login",
]


def evaluate(name, candidate, X_train, X_test, y_train, y_test, X, y, setting, deployed=False):
    """Fit once for test metrics, then cross-validate ROC-AUC on all rows."""
    started = perf_counter()
    candidate.fit(X_train, y_train)
    training_seconds = round(perf_counter() - started, 2)
    fitted = candidate.best_estimator_ if isinstance(candidate, GridSearchCV) else candidate
    used_setting = setting(candidate) if callable(setting) else setting
    predicted = fitted.predict(X_test)
    probabilities = fitted.predict_proba(X_test)[:, 1]
    cv_roc_auc = cross_val_score(fitted, X, y, cv=5, scoring="roc_auc").mean()
    return {
        "name": name,
        "deployed": deployed,
        "setting": used_setting,
        "accuracy": round(float(accuracy_score(y_test, predicted)), 3),
        "precision": round(float(precision_score(y_test, predicted, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, predicted, zero_division=0)), 3),
        "f1": round(float(f1_score(y_test, predicted, zero_division=0)), 3),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 3),
        "cv_roc_auc": round(float(cv_roc_auc), 3),
        "training_seconds": training_seconds,
    }


def main():
    students = pd.read_csv(ROOT / "students.csv")
    X, y = students[FEATURES], students["at_risk"]
    # This exactly matches the split used by train_model.py.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    metrics = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
    best_params = metrics["best_params"]

    logistic = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    neighbors = GridSearchCV(
        make_pipeline(StandardScaler(), KNeighborsClassifier()),
        {"kneighborsclassifier__n_neighbors": [3, 5, 7, 11, 15, 21]},
        cv=5, scoring="recall",
    )
    tree = GridSearchCV(
        DecisionTreeClassifier(class_weight="balanced", random_state=42),
        {"max_depth": [3, 5, 7, 10]}, cv=5, scoring="recall",
    )
    forest = RandomForestClassifier(class_weight="balanced", random_state=42, **best_params)
    candidates = [
        ("Logistic Regression", logistic, "class_weight=balanced", False),
        ("K-Nearest Neighbors", neighbors, lambda fitted: f"k={int(fitted.best_params_['kneighborsclassifier__n_neighbors'])}", False),
        ("Decision Tree", tree, lambda fitted: f"max_depth={int(fitted.best_params_['max_depth'])}", False),
        ("Random Forest", forest, ", ".join(f"{key}={value}" for key, value in best_params.items()), True),
    ]
    results = [
        evaluate(name, candidate, X_train, X_test, y_train, y_test, X, y, setting, deployed)
        for name, candidate, setting, deployed in candidates
    ]
    report = {"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "models": results}
    with (ROOT / "comparison.json").open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)

    print(f"{'Model':23} {'Accuracy':>8} {'Precision':>9} {'Recall':>7} {'F1':>7} {'ROC-AUC':>8} {'CV AUC':>7} {'Time(s)':>8}")
    for item in results:
        print(f"{item['name']:23} {item['accuracy']:8.3f} {item['precision']:9.3f} {item['recall']:7.3f} {item['f1']:7.3f} {item['roc_auc']:8.3f} {item['cv_roc_auc']:7.3f} {item['training_seconds']:8.2f}")
    print("Saved comparison.json")


if __name__ == "__main__":
    main()
