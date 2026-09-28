"""Counterfactual explanations using the saved model's predict_proba method."""

import pandas as pd


def load_reference(csv_path, features):
    """Find a typical on-track profile once from the sample data."""
    if not csv_path.is_file():
        return None
    students = pd.read_csv(csv_path)
    on_track = students.loc[students["at_risk"] == 0, features]
    if on_track.empty:
        return None
    return on_track.median().to_dict()


def feature_wording(name, value):
    """Put a student's feature value into short, readable words."""
    number = f"{value:g}"
    wording = {
        "avg_quiz_score": f"low average quiz score ({number})",
        "avg_assignment_score": f"low average assignment score ({number})",
        "late_submission_rate": f"{value:.0%} of submissions were late",
        "logins_per_week": f"infrequent logins ({number} per week)",
        "avg_session_min": f"short sessions ({number} minutes on average)",
        "missed_deadlines": f"{number} missed deadlines",
        "forum_posts": f"little forum participation ({number} posts)",
        "days_since_last_login": f"{number} days since the last login",
    }
    return wording[name]


def explain_batch(model, students, features, reference, baseline):
    """Compare each feature with its on-track median in eight batch calls."""
    if reference is None or students.empty:
        return [{"drivers": [], "strengths": []} for _ in range(len(students))]
    rows = students[features].reset_index(drop=True)
    contributions = {}
    for name in features:
        modified = rows.copy()
        modified[name] = reference[name]
        # A positive drop means this student's value raised the risk score.
        contributions[name] = (baseline - model.predict_proba(modified)[:, 1]) * 100

    explanations = []
    for index, student in rows.iterrows():
        drivers = sorted(
            ((name, float(contributions[name][index])) for name in features if contributions[name][index] >= 2),
            key=lambda item: item[1], reverse=True,
        )[:3]
        strengths = sorted(
            ((name, float(contributions[name][index])) for name in features if contributions[name][index] <= -2),
            key=lambda item: item[1],
        )[:2]
        explanations.append({
            "drivers": [
                {"text": feature_wording(name, float(student[name])), "points": round(points, 1)}
                for name, points in drivers
            ],
            "strengths": [
                {"text": name.replace("_", " "), "points": round(-points, 1)}
                for name, points in strengths
            ],
        })
    return explanations
