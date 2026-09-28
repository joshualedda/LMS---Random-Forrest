"""Generate a simulated LMS student dataset for training.

Each row = one student in one course. Features mirror what the LMS logs
(attendance_logs, submissions, quiz_results). The at_risk label comes from a
hidden risk score plus noise, so the model has real patterns to learn but is
not perfectly predictable.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1500

logins_per_week = np.clip(rng.normal(4, 2, N), 0, 14).round(1)
avg_session_min = np.clip(rng.normal(35, 15, N), 2, 120).round(1)
avg_quiz_score = np.clip(rng.normal(72, 15, N), 0, 100).round(1)
avg_assignment_score = np.clip(avg_quiz_score + rng.normal(0, 8, N), 0, 100).round(1)
late_submission_rate = np.clip(rng.beta(2, 6, N), 0, 1).round(2)
missed_deadlines = rng.poisson(1.5 + late_submission_rate * 5)
forum_posts = rng.poisson(np.clip(logins_per_week * 0.8, 0.1, None))
days_since_last_login = np.clip(rng.exponential(4, N) - logins_per_week * 0.3, 0, 40).round(0)

# Hidden risk score (higher = more likely at risk)
risk = (
    -0.35 * logins_per_week
    - 0.03 * avg_session_min
    - 0.05 * avg_quiz_score
    - 0.03 * avg_assignment_score
    + 4.0 * late_submission_rate
    + 0.35 * missed_deadlines
    - 0.05 * forum_posts
    + 0.12 * days_since_last_login
)
risk = (risk - risk.mean()) / risk.std()
prob = 1 / (1 + np.exp(-(1.6 * risk - 0.9 + rng.normal(0, 0.5, N))))
at_risk = (rng.random(N) < prob).astype(int)

df = pd.DataFrame({
    "logins_per_week": logins_per_week,
    "avg_session_min": avg_session_min,
    "avg_quiz_score": avg_quiz_score,
    "avg_assignment_score": avg_assignment_score,
    "late_submission_rate": late_submission_rate,
    "missed_deadlines": missed_deadlines,
    "forum_posts": forum_posts,
    "days_since_last_login": days_since_last_login,
    "at_risk": at_risk,
})
df.to_csv("students.csv", index=False)
print(f"Saved students.csv with {len(df)} rows; at-risk share: {df.at_risk.mean():.1%}")
