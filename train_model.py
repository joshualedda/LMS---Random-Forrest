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

# Keep training files beside this script regardless of the current folder.
PROJECT_DIR = Path(__file__).resolve().parent
df = pd.read_csv(PROJECT_DIR / "students.csv")
X, y = df[FEATURES], df["at_risk"]

# Step 4: split (80/20, stratified so both sets keep the same at-risk share)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5 + 8: train with light hyperparameter tuning (5-fold cross-validation)
grid = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    param_grid={
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5, 10],
    },
    scoring="recall",  # catching at-risk students matters most
    cv=5,
    n_jobs=-1,
)
grid.fit(X_train, y_train)
model = grid.best_estimator_
print("Best params:", grid.best_params_)

# Step 6: evaluate on unseen data
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
matrix = confusion_matrix(y_test, pred, labels=[0, 1])
roc_auc = roc_auc_score(y_test, proba)
cv_accuracy = cross_val_score(model, X, y, cv=5).mean()
print("\nConfusion matrix:\n", matrix)
print("\n", classification_report(y_test, pred, target_names=["not at risk", "at risk"]))
print("ROC-AUC:", round(roc_auc, 3))
print("5-fold CV accuracy:", round(cv_accuracy, 3))

# Step 7: feature importance
print("\nFeature importance:")
sorted_importances = sorted(zip(FEATURES, model.feature_importances_), key=lambda t: -t[1])
for name, imp in sorted_importances:
    print(f"  {name:24s} {imp:.3f}")

# Convert NumPy and scikit-learn values to plain JSON numbers.
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
    },
    "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}

# Preserve the existing model bundle format used by Predict and Dashboard.
joblib.dump({"model": model, "features": FEATURES}, PROJECT_DIR / "model.pkl")
print("\nSaved model.pkl")
with (PROJECT_DIR / "metrics.json").open("w", encoding="utf-8") as output:
    json.dump(metrics, output, indent=2)
print("Saved metrics.json")
