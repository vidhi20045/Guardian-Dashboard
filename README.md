# Disaster Prediction & Fairness-Aware Allocation — Dashboard

Interactive supplement to the IJDRR submission. Five tabs: county risk prediction,
fair resource allocation, climate stress testing, results summary, and export.

## Run locally

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Opens at http://localhost:8501.

## Data

The dashboard reads pre-computed results from `data/` (no training at runtime):

| File | Contents | Source |
|---|---|---|
| `metrics_summary.json` | All headline metrics | real project results |
| `allocation_results.csv` | Supply × method comparison | real (step13/step27) |
| `climate_stress.csv` | IPCC scenario coverage | real (step31) |
| `frontier.csv` | Fairness–efficiency frontier | real (step20) |
| `county_predictions.csv` | Per-county risk + CI + SHAP | **placeholder** until regenerated |

### Generate real per-county predictions

The bundled `county_predictions.csv` is illustrative. To populate it with your
trained model's real output:

```bash
# from the project root (where data/processed/model_table_withzones.csv lives)
python make_dashboard_data.py
# then copy the new data/county_predictions.csv into this dashboard's data/ folder
```

This writes each county's latest-month risk, a 3-seed ensemble 90% interval, and
its top-3 SHAP drivers. (County display names default to FIPS codes; swap in a
FIPS→name lookup if you have one. Map location uses state centroids as an
approximation.)

## Publication figures

Every Plotly chart exports a 300-DPI PNG via the camera icon that appears top-right
when you hover over it.

## Deploy (optional)

Push to GitHub and connect the repo at https://share.streamlit.io to get a public
URL for the paper supplement. No authentication; static data; reproducible.
