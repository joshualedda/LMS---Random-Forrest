"""Train and evaluate the Random Forest at-risk student classifier."""
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

FEATURES = [
    "logins_per_week", "avg_session_min", "avg_quiz_score",
    "avg_assignment_score", "late_submission_rate", "missed_deadlines",
    "forum_posts", "days_since_last_login",
]
PROJECT_DIR = Path(__file__).resolve().parent


def train_and_evaluate(df, feature_columns=FEATURES):
    """Train on labelled rows and return a deployable bundle and JSON metrics."""
    X, y = df[feature_columns], df["at_risk"]
    # The same stratified split and tuning grid are used by CLI and web retraining.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        param_grid={
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5, 10],
        },
        scoring="recall", cv=5, n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    matrix = confusion_matrix(y_test, pred, labels=[0, 1])
    roc_auc = roc_auc_score(y_test, proba)
    cv_accuracy = cross_val_score(model, X, y, cv=5).mean()
    sorted_importances = sorted(
        zip(feature_columns, model.feature_importances_), key=lambda item: -item[1]
    )
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 3),
        "precision": round(float(precision_score(y_test, pred)), 3),
        "recall": round(float(recall_score(y_test, pred)), 3),
        "f1": round(float(f1_score(y_test, pred)), 3),
        "roc_auc": round(float(roc_auc), 3),
        "cv_accuracy": round(float(cv_accuracy), 3),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "feature_importances": [
            {"feature": name, "importance": float(importance)}
            for name, importance in sorted_importances
        ],
        "best_params": {
            name: value.item() if hasattr(value, "item") else value
            for name, value in grid.best_params_.items()
        },
        "dataset": {
            "total_rows": int(len(df)),
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "at_risk_share_pct": round(float(y.mean() * 100), 1),
            "class_0": int((y == 0).sum()),
            "class_1": int((y == 1).sum()),
        },
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    # Store the on-track medians with the model so explanations use its data.
    reference = df.loc[y == 0, feature_columns].median().to_dict()
    bundle = {"model": model, "features": list(feature_columns), "reference_profile": reference}
    return bundle, metrics


def main():
    df = pd.read_csv(PROJECT_DIR / "students.csv")
    bundle, metrics = train_and_evaluate(df, FEATURES)
    metrics["trained_on"] = "simulated students.csv"
    print("Best params:", metrics["best_params"])
    matrix = metrics["confusion_matrix"]
    print("\nConfusion matrix:\n", [[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]])
    # Keep the original CLI's evaluation detail.
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["at_risk"], test_size=0.2, random_state=42, stratify=df["at_risk"]
    )
    print("\n", classification_report(y_test, bundle["model"].predict(X_test), target_names=["not at risk", "at risk"]))
    print("ROC-AUC:", metrics["roc_auc"])
    print("5-fold CV accuracy:", metrics["cv_accuracy"])
    print("\nFeature importance:")
    for item in metrics["feature_importances"]:
        print(f"  {item['feature']:24s} {item['importance']:.3f}")
    joblib.dump(bundle, PROJECT_DIR / "model.pkl")
    print("\nSaved model.pkl")
    with (PROJECT_DIR / "metrics.json").open("w", encoding="utf-8") as output:
        json.dump(metrics, output, indent=2)
    print("Saved metrics.json")


if __name__ == "__main__":
    main()
