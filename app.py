"""Single-student prediction page for the LMS prototype."""

from decimal import Decimal, InvalidOperation
from datetime import date, datetime, timezone
from functools import wraps
from pathlib import Path
from io import BytesIO, StringIO
import csv
import json
import math
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request, send_file, session, url_for
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from werkzeug.security import check_password_hash
try:
    import config  # Local, git-ignored credentials; start from config.example.py.
except ModuleNotFoundError as exc:
    if exc.name == "config":
        raise RuntimeError("Create config.py from config.example.py before starting EduGuard.") from exc
    raise
from explain import explain_batch, load_reference
from train_model import FEATURES, train_and_evaluate


app = Flask(__name__)
# A persistent local key keeps development sessions valid across restarts.
PROJECT_DIR = Path(__file__).resolve().parent
SECRET_PATH = PROJECT_DIR / ".secret_key"
if os.environ.get("SECRET_KEY"):
    app.secret_key = os.environ["SECRET_KEY"]
else:
    try:
        with SECRET_PATH.open("x", encoding="utf-8") as output:
            output.write(secrets.token_hex(32))
    except FileExistsError:
        pass
    app.secret_key = SECRET_PATH.read_text(encoding="utf-8").strip()
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_UPLOAD_ROWS = 5000

# Resolve the model beside this file, even when Flask starts elsewhere.
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
STUDENTS_PATH = Path(__file__).resolve().parent / "students.csv"
METRICS_PATH = Path(__file__).resolve().parent / "metrics.json"
COMPARISON_PATH = Path(__file__).resolve().parent / "comparison.json"
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
HISTORY_DIR = Path(__file__).resolve().parent / "models" / "history"
model_lock = threading.Lock()
bundle = joblib.load(MODEL_PATH) if MODEL_PATH.is_file() else None
model = bundle["model"] if bundle is not None else None
features = bundle["features"] if bundle is not None else None
reference_profile = bundle.get("reference_profile") if bundle is not None else None
if reference_profile is None and features is not None:
    reference_profile = load_reference(STUDENTS_PATH, features)  # Legacy model.pkl
model_changed = False


def login_required(view):
    """Send anonymous visitors to login and remember their requested GET page."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            if request.method == "GET":
                session["next_page"] = request.full_path.rstrip("?")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped

# Form rules and example values are kept together so the page is easy to update.
FIELDS = [
    {"name": "logins_per_week", "label": "Logins per week", "help": "Average logins per week, 0 to 14.", "min": 0, "max": 14, "step": "any", "example": "6"},
    {"name": "avg_session_min", "label": "Average session (minutes)", "help": "Average session length, 0 to 120 minutes.", "min": 0, "max": 120, "step": "any", "example": "45"},
    {"name": "avg_quiz_score", "label": "Average quiz score", "help": "Average quiz score, 0 to 100.", "min": 0, "max": 100, "step": "any", "example": "78"},
    {"name": "avg_assignment_score", "label": "Average assignment score", "help": "Average assignment score, 0 to 100.", "min": 0, "max": 100, "step": "any", "example": "82"},
    {"name": "late_submission_rate", "label": "Late submission rate", "help": "Share of submissions that were late, 0 to 1.", "min": 0, "max": 1, "step": "any", "example": "0.1"},
    {"name": "missed_deadlines", "label": "Missed deadlines", "help": "Whole number of missed deadlines, 0 or more.", "min": 0, "max": None, "step": "1", "example": "1", "integer": True},
    {"name": "forum_posts", "label": "Forum posts", "help": "Whole number of forum posts, 0 or more.", "min": 0, "max": None, "step": "1", "example": "4", "integer": True},
    {"name": "days_since_last_login", "label": "Days since last login", "help": "Days since the student last logged in, 0 or more.", "min": 0, "max": None, "step": "any", "example": "2"},
]

MESSAGES = {
    "low": "This student shows few warning signs. Keep supporting their progress.",
    "medium": "This student shows some warning signs. Consider checking in.",
    "high": "This student shows several warning signs. Consider reaching out.",
}


def validate_form(form):
    """Check every field and return parsed values, errors, and typed text."""
    values, errors, typed = {}, {}, {}
    for field in FIELDS:
        name = field["name"]
        raw = form.get(name, "")
        typed[name] = raw
        if not raw.strip():
            errors[name] = "This field is required."
            continue

        try:
            number = Decimal(raw.strip())
            if not number.is_finite():
                raise InvalidOperation
        except InvalidOperation:
            errors[name] = "Enter a valid number."
            continue

        if field.get("integer") and number != number.to_integral_value():
            errors[name] = "Enter a whole number."
        elif number < field["min"] or (field["max"] is not None and number > field["max"]):
            upper = f" to {field['max']}" if field["max"] is not None else " or more"
            errors[name] = f"Enter a value from {field['min']}{upper}."
        else:
            try:
                # Reject values too large for the model's numeric input.
                finite_value = float(number)
                if not math.isfinite(finite_value):
                    raise ValueError
                parsed = int(number) if field.get("integer") else finite_value
            except (OverflowError, ValueError):
                errors[name] = "Enter a valid number."
            else:
                values[name] = parsed
    return values, errors, typed


def predict_one(student):
    """Return risk details for one validated student record."""
    # The saved feature list fixes the DataFrame column order used in training.
    row = pd.DataFrame([[student[name] for name in features]], columns=features)
    probability = float(model.predict_proba(row)[0, 1])
    level = "high" if probability >= 0.66 else "medium" if probability >= 0.33 else "low"
    explanation = explain_batch(model, row, features, reference_profile, np.array([probability]))[0]
    return {
        "level": level,
        "percentage": f"{probability:.1%}",
        "message": MESSAGES[level],
        "explanation": explanation,
    }


def add_explanations(students):
    """Attach reusable explanation text to scored rows."""
    explanations = explain_batch(
        model, students, features, reference_profile,
        students["risk_probability"].to_numpy(),
    )
    students["explanation_json"] = [json.dumps(item) for item in explanations]
    students["main_risk_factors"] = [
        ", ".join(driver["text"] for driver in item["drivers"][:2])
        for item in explanations
    ]
    return students


def score_students(students):
    """Score a validated DataFrame in one model call."""
    students = students.copy()
    if "student_id" not in students:
        students["student_id"] = [f"S{number:03d}" for number in range(1, len(students) + 1)]
    if students.empty:
        students["risk_probability"] = pd.Series(dtype=float)
        students["risk_level"] = pd.Series(dtype=str)
        return add_explanations(students)

    # Select saved model features in their training order before batch prediction.
    students["risk_probability"] = model.predict_proba(students[features])[:, 1]
    students["risk_level"] = np.select(
        [students["risk_probability"] >= 0.66, students["risk_probability"] >= 0.33],
        ["high", "medium"], default="low",
    )
    return add_explanations(students)


def dashboard_view(students, uploaded=False):
    """Build cards, sorted rows, and optional label comparison from scored data."""
    counts = students["risk_level"].value_counts()
    summary = {
        "total": len(students),
        "high": int(counts.get("high", 0)),
        "medium": int(counts.get("medium", 0)),
        "low": int(counts.get("low", 0)),
        "average": f"{students['risk_probability'].mean():.1%}" if len(students) else "0.0%",
    }
    comparison = None
    if uploaded and "at_risk" in students and len(students):
        # Labels use 0.5 for classification; badges still use the three risk bands.
        actual = students["at_risk"].astype(int)
        predicted = (students["risk_probability"] >= 0.5).astype(int)
        tn, fp, fn, tp = (int(value) for value in confusion_matrix(actual, predicted, labels=[0, 1]).ravel())
        comparison = {
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "accuracy": f"{accuracy_score(actual, predicted):.1%}",
            "precision": f"{precision_score(actual, predicted, zero_division=0):.1%}",
            "recall": f"{recall_score(actual, predicted, zero_division=0):.1%}",
            "f1": f"{f1_score(actual, predicted, zero_division=0):.1%}",
        }
    ordered = students.sort_values("risk_probability", ascending=False, kind="stable").copy()
    ordered["risk_percent"] = ordered["risk_probability"].map(lambda value: f"{value:.1%}")
    rows = ordered.to_dict("records")
    for row in rows:
        row["explanation"] = json.loads(row["explanation_json"])
    return {"rows": rows, "summary": summary, "comparison": comparison}


def clean_uploaded_data(content, invalid_action):
    """Validate one CSV and return (clean DataFrame or None, messages)."""
    messages = []
    try:
        text = content.decode("utf-8-sig")
        headers = next(csv.reader(StringIO(text)))
        names = [name.strip().lower() for name in headers]
        if len(names) != len(set(names)):
            return None, ["Column names must be unique after spaces and letter case are ignored."]
        students = pd.read_csv(StringIO(text), dtype=str, keep_default_na=False, skip_blank_lines=False)
    except (UnicodeDecodeError, StopIteration, csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None, ["The file could not be read as a UTF-8 CSV. Check its format and try again."]

    students.columns = names
    required = [field["name"] for field in FIELDS]
    missing = [name for name in required if name not in students]
    if missing:
        return None, ["Missing required columns: " + ", ".join(missing) + "."]
    if len(students) > MAX_UPLOAD_ROWS:
        return None, [f"The file has {len(students)} rows. The maximum is {MAX_UPLOAD_ROWS:,}."]
    if students.empty:
        return None, ["The CSV has no student rows."]

    # Collect bad rows across all required numeric fields before skipping or rejecting.
    bad_rows = pd.Series(False, index=students.index)
    numbers = {}
    for field in FIELDS:
        name = field["name"]
        values = pd.to_numeric(students[name].str.strip(), errors="coerce")
        invalid = ~np.isfinite(values.to_numpy(dtype=float))
        if field.get("integer"):
            invalid |= (values % 1 != 0).fillna(True).to_numpy()
        bad_rows |= invalid
        numbers[name] = values
    if "at_risk" in students:
        labels = pd.to_numeric(students["at_risk"].str.strip(), errors="coerce")
        bad_rows |= ~labels.isin([0, 1])
        numbers["at_risk"] = labels

    bad_count = int(bad_rows.sum())
    if bad_count:
        examples = ", ".join(str(index + 2) for index in students.index[bad_rows][:5])
        messages.append(f"{bad_count} row(s) have missing, non-numeric, or non-whole required values or invalid at_risk labels. First CSV row numbers: {examples}.")
        if invalid_action == "reject":
            messages.append("The file was rejected. Choose Skip those rows to continue without them.")
            return None, messages
        students = students.loc[~bad_rows].copy()
        messages.append(f"Skipped {bad_count} invalid row(s).")
    if students.empty:
        messages.append("No valid student rows remain after skipping.")
        return None, messages

    adjusted = 0
    for field in FIELDS:
        name = field["name"]
        values = numbers[name].loc[students.index]
        clamped = values.clip(lower=field["min"], upper=field["max"])
        adjusted += int((clamped != values).sum())
        students[name] = clamped
    if "at_risk" in students:
        students["at_risk"] = numbers["at_risk"].loc[students.index].astype(int)
    if "student_id" in students:
        ids = students["student_id"].astype(str).str.strip()
        fallback = pd.Series([f"S{index + 1:03d}" for index in students.index], index=students.index)
        students["student_id"] = ids.where(ids != "", fallback)
    else:
        students["student_id"] = [f"S{index + 1:03d}" for index in students.index]
    if adjusted:
        messages.append(f"Adjusted {adjusted} out-of-range value(s) to the nearest valid limit.")
    columns = ["student_id", *required] + (["at_risk"] if "at_risk" in students else [])
    return students[columns].reset_index(drop=True), messages


def remove_old_uploads():
    """Delete only this app's old, randomly named upload CSVs."""
    UPLOAD_DIR.mkdir(exist_ok=True)
    cutoff = time.time() - 3600
    for path in UPLOAD_DIR.glob("*.csv"):
        if re.fullmatch(r"[0-9a-f]{32}\.csv", path.name):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except FileNotFoundError:
                pass  # Another request may have removed the same old file.


def current_upload():
    """Read only the server-generated CSV named by this session's file ID."""
    upload_id = session.get("upload_id")
    if not isinstance(upload_id, str) or not re.fullmatch(r"[0-9a-f]{32}", upload_id):
        return None
    path = UPLOAD_DIR / f"{upload_id}.csv"
    if not path.is_file():
        return None
    try:
        cleaned, _messages = clean_uploaded_data(path.read_bytes(), "reject")
        return cleaned
    except (OSError, ValueError):
        return None


def retrain_eligibility(students):
    """Check labels and enough examples for the stratified 5-fold search."""
    if students is None:
        return "Upload a CSV first to retrain the model."
    if "at_risk" not in students:
        return "Retraining needs an at_risk column with 0 and 1 labels."
    counts = students["at_risk"].value_counts()
    if len(students) < 50:
        return f"Retraining needs at least 50 rows with both outcomes present. Your file has {len(students)} rows."
    if counts.get(0, 0) < 10 or counts.get(1, 0) < 10:
        return ("Retraining needs at least 10 examples of each outcome "
                f"(at_risk=0 and at_risk=1). Your file has {counts.get(0, 0)} and {counts.get(1, 0)}.")
    return None


def latest_backup():
    """Only accept a complete model/metrics pair."""
    for saved_model in sorted(HISTORY_DIR.glob("model_*.pkl"), reverse=True):
        saved_metrics = HISTORY_DIR / saved_model.name.replace("model_", "metrics_").replace(".pkl", ".json")
        if saved_metrics.is_file():
            return saved_model, saved_metrics
    return None


def staged_file(destination, writer):
    """Write a complete file beside its destination before replacing it."""
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".retrain_", delete=False) as handle:
        path = Path(handle.name)
    try:
        writer(path)
        return path
    except Exception:
        path.unlink(missing_ok=True)
        raise


def install_pair(new_model, new_metrics, backup_pair=None):
    """Replace both files; restore the old pair if the second replace fails."""
    rollback_model = rollback_metrics = None
    try:
        if backup_pair is not None:
            rollback_model = staged_file(MODEL_PATH, lambda path: shutil.copy2(backup_pair[0], path))
            rollback_metrics = staged_file(METRICS_PATH, lambda path: shutil.copy2(backup_pair[1], path))
        os.replace(new_model, MODEL_PATH)
        os.replace(new_metrics, METRICS_PATH)
    except Exception:
        if rollback_model is not None and rollback_metrics is not None:
            os.replace(rollback_model, MODEL_PATH)
            os.replace(rollback_metrics, METRICS_PATH)
        raise
    finally:
        new_model.unlink(missing_ok=True)
        new_metrics.unlink(missing_ok=True)
        if rollback_model is not None:
            rollback_model.unlink(missing_ok=True)
        if rollback_metrics is not None:
            rollback_metrics.unlink(missing_ok=True)


def refresh_model():
    """Make Predict, Dashboard, and explanations use the newly active model."""
    global bundle, model, features, reference_profile, dashboard_data, model_changed
    bundle = joblib.load(MODEL_PATH)
    model, features = bundle["model"], bundle["features"]
    reference_profile = bundle.get("reference_profile")
    if reference_profile is None:
        reference_profile = load_reference(STUDENTS_PATH, features)
    dashboard_data = dashboard_view(score_students(pd.read_csv(STUDENTS_PATH))) if STUDENTS_PATH.is_file() else None
    model_changed = True


def active_dashboard_data():
    """Use this session's upload if its saved CSV still exists."""
    upload_id = session.get("upload_id")
    if isinstance(upload_id, str) and re.fullmatch(r"[0-9a-f]{32}", upload_id):
        path = UPLOAD_DIR / f"{upload_id}.csv"
        if path.is_file():
            students = pd.read_csv(path, dtype={"student_id": str})
            if model_changed:
                students = score_students(students.drop(columns=["risk_probability", "risk_level", "explanation_json", "main_risk_factors"], errors="ignore"))
            elif "explanation_json" not in students:
                students = add_explanations(students)
            return dashboard_view(students, uploaded=True), True
    session.pop("upload_id", None)
    return dashboard_data, False


# Cache the sample dataset once at startup; uploaded CSVs are scored on POST.
dashboard_data = dashboard_view(score_students(pd.read_csv(STUDENTS_PATH))) if model is not None and STUDENTS_PATH.is_file() else None


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("predict"))
    error = None
    if request.method == "POST":
        now = time.time()
        started = session.get("login_attempt_started", 0)
        if not isinstance(started, (int, float)) or now - started >= 600:
            session["login_attempt_started"] = now
            session["login_failures"] = 0

        # The signed session holds a ten-minute failure window for this browser.
        if session.get("login_failures", 0) >= 5:
            error = "Too many attempts. Try again in a few minutes."
        elif (request.form.get("username", "") == config.ADMIN_USERNAME
              and check_password_hash(config.ADMIN_PASSWORD_HASH, request.form.get("password", ""))):
            target = session.pop("next_page", url_for("predict"))
            session.pop("login_failures", None)
            session.pop("login_attempt_started", None)
            session["logged_in"] = True
            session["username"] = config.ADMIN_USERNAME
            # Redirect only to a local path remembered by the guard.
            if not isinstance(target, str) or not target.startswith("/") or target.startswith("//") or "\\" in target:
                target = url_for("predict")
            return redirect(target)
        else:
            session["login_failures"] = session.get("login_failures", 0) + 1
            error = "Incorrect username or password"
    return render_template("login.html", title=config.APP_TITLE, error=error,
                           signed_out=request.args.get("signed_out") == "1")


@app.post("/logout")
@login_required
def logout():
    session.clear()
    return redirect(url_for("login", signed_out=1))


@app.route("/")
@login_required
def home():
    return redirect(url_for("predict"))


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    # Render an actionable message if training has not produced model.pkl yet.
    if model is None:
        return render_template("predict.html", model_missing=True), 503

    typed = {field["name"]: field["example"] for field in FIELDS}
    errors = {}
    result = None
    if request.method == "POST":
        values, errors, typed = validate_form(request.form)
        if not errors:
            result = predict_one(values)

    return render_template(
        "predict.html", fields=FIELDS, typed=typed, errors=errors, result=result,
        model_missing=False,
    )


@app.route("/dashboard")
@login_required
def dashboard():
    data, uploaded = active_dashboard_data()
    if data is None:
        return render_template("dashboard.html", missing="model" if model is None else "csv"), 503

    level = request.args.get("level", "all").lower()
    if level not in ("all", "high", "medium", "low"):
        level = "all"
    rows = data["rows"]
    filtered = rows if level == "all" else [row for row in rows if row["risk_level"] == level]

    # Page only the filtered list, but keep summary cards for the full CSV.
    page_count = max(1, math.ceil(len(filtered) / 25))
    page = min(max(request.args.get("page", 1, type=int), 1), page_count)
    visible = filtered[(page - 1) * 25:page * 25]
    return render_template(
        "dashboard.html", missing=None, summary=data["summary"],
        students=visible, level=level, page=page, page_count=page_count,
        filtered_count=len(filtered), uploaded=uploaded, comparison=data["comparison"],
        notice=request.args.get("notice", "") if uploaded else "",
    )


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    # Limit only this route so existing Predict requests keep their behavior.
    request.max_content_length = MAX_UPLOAD_BYTES + 64 * 1024
    errors = []
    if request.method == "POST":
        try:
            remove_old_uploads()
        except OSError:
            return render_template("upload.html", fields=FIELDS, errors=["Upload storage is unavailable. Please try again later."]), 503
        uploaded_file = request.files.get("file")
        invalid_action = request.form.get("invalid_action", "skip")
        if invalid_action not in ("skip", "reject"):
            invalid_action = "skip"
        if model is None:
            errors = ["The model is missing. Run python train_model.py, then restart the app."]
        elif uploaded_file is None or not uploaded_file.filename:
            errors = ["Choose a CSV file to upload."]
        elif not uploaded_file.filename.lower().endswith(".csv"):
            errors = ["Choose a file ending in .csv."]
        else:
            content = uploaded_file.stream.read(MAX_UPLOAD_BYTES + 1)
            if len(content) > MAX_UPLOAD_BYTES:
                errors = ["The file is too large. Upload a CSV of 2 MB or less."]
            else:
                cleaned, messages = clean_uploaded_data(content, invalid_action)
                if cleaned is None:
                    errors = messages
                else:
                    # Save only a server-generated name; the session holds only its ID.
                    scored = score_students(cleaned)
                    upload_id = uuid4().hex
                    path = UPLOAD_DIR / f"{upload_id}.csv"
                    try:
                        scored.to_csv(path, index=False)
                    except OSError:
                        errors = ["The file could not be saved. Please try again later."]
                        return render_template("upload.html", fields=FIELDS, errors=errors), 503
                    session["upload_id"] = upload_id
                    return redirect(url_for("dashboard", notice=" ".join(messages)))
    upload_data = current_upload()
    return render_template(
        "upload.html", fields=FIELDS, errors=errors,
        retrain_ready=upload_data is not None and retrain_eligibility(upload_data) is None,
        retrain_message=retrain_eligibility(upload_data) if upload_data is not None else None,
        notice=request.args.get("notice", ""),
    )


@app.route("/retrain", methods=["GET", "POST"])
@login_required
def retrain():
    students = current_upload()
    reason = retrain_eligibility(students)
    if reason:
        return redirect(url_for("upload", notice=reason))
    counts = students["at_risk"].value_counts()
    if request.method == "GET":
        return render_template("retrain.html", rows=len(students), class_0=int(counts[0]), class_1=int(counts[1]))

    with model_lock:
        new_model = new_metrics = None
        installed = False
        backup_model = backup_metrics = None
        try:
            # Train and serialize fully before touching the working pair.
            new_bundle, metrics = train_and_evaluate(students, FEATURES)
            upload_path = UPLOAD_DIR / f"{session['upload_id']}.csv"
            uploaded_at = datetime.fromtimestamp(upload_path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d")
            metrics["trained_on"] = f"uploaded file, uploaded {uploaded_at} (UTC)"
            new_model = staged_file(MODEL_PATH, lambda path: joblib.dump(new_bundle, path))
            new_metrics = staged_file(METRICS_PATH, lambda path: path.write_text(json.dumps(metrics, indent=2), encoding="utf-8"))

            HISTORY_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%f")
            backup_model = HISTORY_DIR / f"model_{stamp}.pkl"
            backup_metrics = HISTORY_DIR / f"metrics_{stamp}.json"
            shutil.copy2(MODEL_PATH, backup_model)
            try:
                shutil.copy2(METRICS_PATH, backup_metrics)
            except Exception:
                backup_model.unlink(missing_ok=True)
                raise
            previous = json.loads(backup_metrics.read_text(encoding="utf-8"))
            install_pair(new_model, new_metrics, (backup_model, backup_metrics))
            installed = True
            refresh_model()
            # Keep the newest five complete backup pairs.
            for old_model in sorted(HISTORY_DIR.glob("model_*.pkl"), reverse=True)[5:]:
                old_metrics = HISTORY_DIR / old_model.name.replace("model_", "metrics_").replace(".pkl", ".json")
                try:
                    old_model.unlink(missing_ok=True)
                    old_metrics.unlink(missing_ok=True)
                except OSError:
                    app.logger.warning("Could not prune old model backup %s", old_model)
        except Exception:
            if installed:
                # A failure while refreshing the app must not activate a half-ready model.
                os.replace(staged_file(MODEL_PATH, lambda path: shutil.copy2(backup_model, path)), MODEL_PATH)
                os.replace(staged_file(METRICS_PATH, lambda path: shutil.copy2(backup_metrics, path)), METRICS_PATH)
                refresh_model()
            if new_model is not None:
                new_model.unlink(missing_ok=True)
            if new_metrics is not None:
                new_metrics.unlink(missing_ok=True)
            app.logger.exception("Retraining failed")
            return render_template("retrain.html", rows=len(students), class_0=int(counts[0]), class_1=int(counts[1]),
                                   error="Retraining could not finish. The previous model is still available. Please check your data and try again."), 500

    labels = [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
              ("F1", "f1"), ("ROC-AUC", "roc_auc"), ("5-fold CV accuracy", "cv_accuracy")]
    comparison = [
        {"label": label, "before": previous[key], "after": metrics[key],
         "arrow": "↑" if metrics[key] > previous[key] else "↓" if metrics[key] < previous[key] else "→"}
        for label, key in labels
    ]
    return render_template("retrain_result.html", metrics=metrics, previous=previous, comparison=comparison)


@app.post("/revert")
@login_required
def revert():
    with model_lock:
        saved = latest_backup()
        if saved is None:
            return redirect(url_for("model_performance", notice="No previous model backup is available."))
        new_model = new_metrics = None
        try:
            new_model = staged_file(MODEL_PATH, lambda path: shutil.copy2(saved[0], path))
            new_metrics = staged_file(METRICS_PATH, lambda path: shutil.copy2(saved[1], path))
            # Keep a rollback copy of the currently active pair if restoration fails.
            rollback_model = staged_file(MODEL_PATH, lambda path: shutil.copy2(MODEL_PATH, path))
            rollback_metrics = staged_file(METRICS_PATH, lambda path: shutil.copy2(METRICS_PATH, path))
            try:
                install_pair(new_model, new_metrics, (rollback_model, rollback_metrics))
                try:
                    refresh_model()
                except Exception:
                    os.replace(rollback_model, MODEL_PATH)
                    os.replace(rollback_metrics, METRICS_PATH)
                    refresh_model()
                    raise
            finally:
                rollback_model.unlink(missing_ok=True)
                rollback_metrics.unlink(missing_ok=True)
        except Exception:
            if new_model is not None:
                new_model.unlink(missing_ok=True)
            if new_metrics is not None:
                new_metrics.unlink(missing_ok=True)
            app.logger.exception("Restore failed")
            return redirect(url_for("model_performance", notice="Could not restore the backup. The active model was kept."))
    return redirect(url_for("model_performance", notice="Previous model restored from backup."))


@app.errorhandler(413)
def upload_too_large(_error):
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("upload.html", fields=FIELDS, errors=["The file is too large. Upload a CSV of 2 MB or less."]), 413


@app.post("/use-sample-data")
@login_required
def use_sample_data():
    session.pop("upload_id", None)
    return redirect(url_for("dashboard"))


@app.get("/sample.csv")
@login_required
def sample_csv():
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["student_id", *[field["name"] for field in FIELDS], "at_risk"])
    # Include both engaged and struggling examples in a small starter CSV.
    for number in range(1, 21):
        if number <= 10:
            values = [11, 75, 85 + number % 10, 88, 0.05, 0, 6, 1, 0]
        else:
            values = [2, 15, 42 + number % 8, 48, 0.75, 5, 0, 12, 1]
        writer.writerow([f"S{number:03d}", *values])
    return send_file(BytesIO(output.getvalue().encode("utf-8")), mimetype="text/csv", as_attachment=True, download_name="sample.csv")


@app.get("/download-results")
@login_required
def download_results():
    data, _uploaded = active_dashboard_data()
    if data is None:
        return redirect(url_for("dashboard"))
    level = request.args.get("level", "all").lower()
    if level not in ("all", "high", "medium", "low"):
        level = "all"
    rows = data["rows"] if level == "all" else [row for row in data["rows"] if row["risk_level"] == level]
    columns = ["student_id", *[field["name"] for field in FIELDS], "risk_probability", "risk_level", "main_risk_factors"]
    if data["rows"] and "at_risk" in data["rows"][0]:
        columns.append("at_risk")
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(columns)
    for student in rows:
        values = [
            f"{student['risk_probability']:.1%}" if name == "risk_probability"
            else ", ".join(driver["text"] for driver in student["explanation"]["drivers"])
            if name == "main_risk_factors" else student[name]
            for name in columns
        ]
        # Prefix formula-like IDs so spreadsheet apps treat them as text.
        if str(values[0]).lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
            values[0] = "'" + str(values[0])
        writer.writerow(values)
    return send_file(
        BytesIO(output.getvalue().encode("utf-8-sig")), mimetype="text/csv",
        as_attachment=True, download_name=f"risk_report_{date.today().isoformat()}.csv",
    )


@app.get("/report")
@login_required
def report():
    data, uploaded = active_dashboard_data()
    if data is None:
        return redirect(url_for("dashboard"))
    metrics = None
    if METRICS_PATH.is_file():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    total = data["summary"]["total"]
    distribution = {
        level: (data["summary"][level] / total * 100 if total else 0)
        for level in ("high", "medium", "low")
    }
    return render_template(
        "report.html", date_generated=date.today().isoformat(),
        source="your uploaded file" if uploaded else "sample data (students.csv)",
        summary=data["summary"], distribution=distribution,
        students=data["rows"][:20], metrics=metrics,
    )


@app.route("/model-performance")
@login_required
def model_performance():
    # Read the training report when requested so retraining updates the page.
    if not METRICS_PATH.is_file():
        return render_template("model_performance.html", missing="missing"), 503
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return render_template("model_performance.html", missing="invalid"), 503

    explanations = [
        ("Accuracy", "accuracy", "Share of all test students classified correctly."),
        ("Precision", "precision", "Of students flagged at risk, this share really were at risk."),
        ("Recall", "recall", "Of students who really were at risk, this share was caught."),
        ("F1", "f1", "A combined score balancing precision and recall."),
        ("ROC-AUC", "roc_auc", "How well the model ranks at-risk students above others."),
    ]
    cards = [
        {"label": label, "value": f"{metrics[key]:.1%}" if key != "roc_auc" else f"{metrics[key]:.3f}", "help": help_text}
        for label, key, help_text in explanations
    ]
    bars = [
        {
            "name": item["feature"].replace("_", " ").capitalize(),
            "percentage": f"{item['importance']:.1%}",
            "width": min(100, max(0, float(item["importance"]) * 100)),
        }
        for item in sorted(
            metrics["feature_importances"],
            key=lambda item: item["importance"],
            reverse=True,
        )
    ]
    return render_template(
        "model_performance.html", missing=None, cards=cards, bars=bars,
        metrics=metrics, notice=request.args.get("notice", ""),
        has_backup=latest_backup() is not None,
    )


@app.get("/model-comparison")
@login_required
def model_comparison():
    if not COMPARISON_PATH.is_file():
        return render_template("model_comparison.html", missing=True), 503
    try:
        report_data = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return render_template("model_comparison.html", missing=True), 503

    models = report_data["models"]
    deployed = next(item for item in models if item["deployed"])
    labels = {
        "accuracy": "accuracy", "precision": "precision", "recall": "recall",
        "f1": "F1", "roc_auc": "ROC-AUC", "cv_roc_auc": "cross-validated ROC-AUC",
        "training_seconds": "training time",
    }
    takeaways = []
    for key, label in labels.items():
        best = min(item[key] for item in models) if key == "training_seconds" else max(item[key] for item in models)
        winners = [item["name"] for item in models if item[key] == best]
        for item in models:
            if item[key] == best:
                item.setdefault("best_keys", []).append(key)
        gap = (deployed[key] - best) if key == "training_seconds" else (best - deployed[key]) * 100
        unit = "seconds" if key == "training_seconds" else "percentage points"
        takeaways.append(f"Best {label}: {', '.join(winners)}. Random Forest is {gap:.2f} {unit} from the best.")
    return render_template(
        "model_comparison.html", missing=False, models=models,
        takeaways=takeaways, trained_at=report_data["trained_at"],
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
