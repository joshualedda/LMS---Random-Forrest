---
name: lms-risk-project
description: Maintain, extend, or explain this Flask LMS student-risk prediction project, including its saved Random Forest, dashboard, uploads, explanations, metrics, comparison, and exports. Use for work inside this project; not for unrelated Flask or ML apps.
---

# LMS risk project

Use this skill for requests about this repository. Read [System.MD](../../System.MD) when the task needs the system concept, algorithms, data contract, route map, or a project-defense explanation. Inspect the current code before editing; the code and generated artifacts are the source of truth if documentation becomes stale.

## Project invariants

- Keep `model.pkl` as a joblib dictionary with `model` and ordered `features`. Build prediction DataFrames with that feature order. Risk display bands are `< 0.33` low, `< 0.66` medium, and otherwise high; uploaded-label evaluation uses a separate `0.5` binary threshold.
- Treat `students.csv` and `model.pkl` as existing artifacts unless the user requests regeneration. `train_model.py` overwrites `model.pkl` and `metrics.json`; `compare_models.py` writes `comparison.json` without changing the deployed model.
- Keep the single-student, default Dashboard, and session-uploaded Dashboard flows working when extending one of them. Uploaded CSVs live under `uploads/` with UUID filenames; the session stores the ID only.
- Keep explanations in `explain.py`. They compare each feature with the median among simulated `at_risk = 0` rows using eight batch `predict_proba` calls. Present them as model sensitivity, not proven causes.
- Use the existing Flask, pandas, NumPy, scikit-learn, joblib, and standard-library stack unless the user authorizes a dependency change. PDF export currently uses the browser print dialog.

## Check changes

Run focused checks for the affected route or script. When prediction, dashboard, or export logic changes, check the sample and upload sources separately, including active filters and optional `at_risk` labels. Confirm that a training or comparison run did not replace an artifact outside the requested scope. Update `System.MD` when an algorithm or data-flow rule changes.
