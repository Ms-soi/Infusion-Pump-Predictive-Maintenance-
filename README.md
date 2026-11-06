# Smart Machine Failure Prediction System

Streamlit prototype for the final-year project *"Intelligent Predictive Maintenance Decision
Support System for Hospital Infusion Pumps: A Machine Learning Approach Using the Gradient
Boosting Algorithm."*

## Running locally

1. Install dependencies (exact versions are pinned deliberately — see "Troubleshooting" below):
   ```
   pip install -r requirements.txt
   ```
2. Make sure `models/gradient_boosting_model.pkl` and `models/standard_scaler.pkl` are present
   (they are included in this folder).
3. Run the app:
   ```
   streamlit run app.py
   ```
4. Open the URL Streamlit prints in your terminal (usually http://localhost:8501).

## Deploying to Streamlit Community Cloud

Just point the deploy tool at this folder's `app.py`. Two files make deployment reliable:
- `requirements.txt` — pins **exact** package versions (not just minimums), so the platform can't
  silently resolve an incompatible scikit-learn version.
- `runtime.txt` — pins the Python version (3.12) to match the environment the model was verified in.

## Troubleshooting

**"No module named '_loss'" (or similar) when loading the model:**
This means the installed scikit-learn version doesn't match the one the model file was pickled
with — pickled scikit-learn models are not guaranteed compatible across versions. Fix: make sure
`requirements.txt` and `runtime.txt` are deployed exactly as included here (don't loosen the
`==` pins to `>=`). If you retrain the model yourself later, re-save it with
`joblib.dump(...)` using whatever scikit-learn version you deploy with, and update the pin in
`requirements.txt` to match.

**Custom styling looks unstyled / raw HTML shows as text:**
This app intentionally avoids injecting raw HTML via `unsafe_allow_html` — some hosting/preview
environments render it unreliably. All styling is done through native Streamlit components and
`.streamlit/config.toml` instead, which is supported everywhere Streamlit runs.

## What it does

- **Single Prediction** — enter operating parameters (air/process temperature, rotational speed,
  torque, tool wear, machine type) and get an instant failure-probability estimate, plus a SHAP
  waterfall chart explaining exactly why the model reached that result.
- **Batch Prediction (CSV)** — upload a CSV of multiple equipment readings and download predictions
  for all of them at once.
- **Model Performance** — the real, verified test-set metrics, confusion matrix, and feature
  importances for the tuned Gradient Boosting model.
- **About & Safety** — project background, dataset source, and required safety disclaimers for a
  medical-equipment decision-support tool.

## Files

```
webapp/
├── app.py                    # the full application
├── requirements.txt          # exact pinned dependency versions
├── runtime.txt                # pinned Python version for cloud deployment
├── README.md
├── .streamlit/
│   └── config.toml           # native Streamlit theme (colors, font)
├── assets/
│   └── app_logo.png          # original infusion-pump icon used as the app logo
└── models/
    ├── gradient_boosting_model.pkl
    └── standard_scaler.pkl
```
