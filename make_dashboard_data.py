"""
make_dashboard_data.py — generate REAL dashboard data from your trained pipeline.

Produces data/county_predictions.csv with each county's latest-month risk, an
ensemble confidence interval, and its top-3 SHAP drivers. The aggregate result
files (allocation_results.csv, climate_stress.csv, frontier.csv, metrics_summary.json)
already hold your real numbers and are left as-is.

Run from the project root (where data/processed/model_table_withzones.csv lives):
    python make_dashboard_data.py
Then copy data/county_predictions.csv into the dashboard's data/ folder.
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

SRC = "data/processed/model_table_withzones.csv"
OUT = "data/county_predictions.csv"
NOT_FEATURES = ["county_fips", "year", "disaster", "date"]

# approximate state centroids (lat, lon) — fallback location when no gazetteer is available
STATE_CENTROID = {
 "01":(32.8,-86.8),"02":(64.2,-149.5),"04":(34.3,-111.7),"05":(34.9,-92.4),"06":(37.2,-119.7),
 "08":(39.0,-105.5),"09":(41.6,-72.7),"10":(39.0,-75.5),"11":(38.9,-77.0),"12":(28.6,-81.5),
 "13":(32.6,-83.4),"15":(20.3,-156.4),"16":(44.4,-114.6),"17":(40.0,-89.2),"18":(39.9,-86.3),
 "19":(42.0,-93.5),"20":(38.5,-98.4),"21":(37.5,-85.3),"22":(31.0,-92.0),"23":(45.4,-69.2),
 "24":(39.0,-76.8),"25":(42.3,-71.8),"26":(44.3,-85.4),"27":(46.3,-94.3),"28":(32.7,-89.7),
 "29":(38.4,-92.5),"30":(47.0,-109.6),"31":(41.5,-99.8),"32":(39.3,-116.6),"33":(43.7,-71.6),
 "34":(40.2,-74.7),"35":(34.4,-106.1),"36":(42.9,-75.6),"37":(35.6,-79.4),"38":(47.5,-100.5),
 "39":(40.3,-82.8),"40":(35.6,-97.5),"41":(43.9,-120.6),"42":(40.9,-77.8),"44":(41.7,-71.6),
 "45":(33.9,-80.9),"46":(44.4,-100.2),"47":(35.9,-86.4),"48":(31.5,-99.3),"49":(39.3,-111.7),
 "50":(44.1,-72.7),"51":(37.5,-78.9),"53":(47.4,-120.5),"54":(38.6,-80.6),"55":(44.6,-90.0),
 "56":(43.0,-107.6),"72":(18.2,-66.4)}


def load_county_geo():
    """fips -> (name, state_abbr, lat, lon) from the US Census county gazetteer.
    Returns {} if the download is unavailable (then we fall back to FIPS + state centroids)."""
    url = ("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
           "2023_Gazetteer/2023_Gaz_counties_national.zip")
    try:
        import io as _io
        import zipfile
        import requests
        r = requests.get(url, timeout=40); r.raise_for_status()
        z = zipfile.ZipFile(_io.BytesIO(r.content))
        txt = z.read([n for n in z.namelist() if n.endswith(".txt")][0]).decode("latin-1")
        lines = txt.splitlines()
        cols = [c.strip() for c in lines[0].split("\t")]
        ix = {c: i for i, c in enumerate(cols)}
        geo = {}
        for ln in lines[1:]:
            p = ln.split("\t")
            if len(p) <= ix.get("INTPTLONG", 99):
                continue
            fips = p[ix["GEOID"]].strip().zfill(5)
            name = p[ix["NAME"]].strip()
            usps = p[ix["USPS"]].strip()
            try:
                lat, lon = float(p[ix["INTPTLAT"]]), float(p[ix["INTPTLONG"]])
            except ValueError:
                lat = lon = None
            geo[fips] = (name, usps, lat, lon)
        print(f"  loaded names + centroids for {len(geo):,} counties from Census gazetteer")
        return geo
    except Exception as e:
        print(f"  gazetteer download unavailable ({e}); using FIPS codes + state centroids")
        return {}


def main():
    df = pd.read_csv(SRC, dtype={"county_fips": str})
    features = [c for c in df.columns if c not in NOT_FEATURES]
    train = df[df["year"] <= 2018]
    test = df[df["year"] >= 2019].copy()
    spw = (len(train) - train["disaster"].sum()) / max(int(train["disaster"].sum()), 1)

    def mk(seed):
        return XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                             reg_lambda=1.5, gamma=0.1, scale_pos_weight=spw,
                             eval_metric="logloss", n_jobs=-1, random_state=seed)

    print("Training main model + 3-seed ensemble for confidence intervals...")
    main_m = mk(42); main_m.fit(train[features], train["disaster"])
    ens = []
    for s in (1, 2, 3):
        m = mk(s); m.fit(train[features], train["disaster"])
        ens.append(m)

    # latest available month per county in the test period
    test["ym"] = test["year"] * 100 + test.get("month", 1)
    latest = test.sort_values("ym").groupby("county_fips").tail(1).reset_index(drop=True)
    X = latest[features]
    latest["risk"] = main_m.predict_proba(X)[:, 1]
    preds = np.column_stack([m.predict_proba(X)[:, 1] for m in ens])
    latest["ci_low"] = np.percentile(preds, 5, axis=1)
    latest["ci_high"] = np.percentile(preds, 95, axis=1)

    # top-3 SHAP drivers per county
    print("Computing SHAP drivers...")
    try:
        import shap
        expl = shap.TreeExplainer(main_m)
        sv = expl.shap_values(X)
        sv = sv[1] if isinstance(sv, list) else sv
        feat = np.array(features)
        top = np.argsort(-np.abs(sv), axis=1)[:, :3]
        for i in (1, 2, 3):
            latest[f"shap{i}"] = [feat[top[r, i-1]] for r in range(len(latest))]
            latest[f"shap{i}_val"] = [abs(sv[r, top[r, i-1]]) for r in range(len(latest))]
    except Exception as e:
        print(f"  SHAP unavailable ({e}); using global importances.")
        imp = pd.Series(main_m.feature_importances_, index=features).sort_values(ascending=False)
        for i in (1, 2, 3):
            latest[f"shap{i}"] = imp.index[i-1]; latest[f"shap{i}_val"] = imp.iloc[i-1]

    width = latest["ci_high"] - latest["ci_low"]
    latest["confidence"] = np.where(width > 0.05, "Low",
                             np.where((latest["risk"] < 0.15) | (latest["risk"] > 0.4),
                                      "High", "Medium"))
    latest["state_fips"] = latest["county_fips"].str[:2]
    geo = load_county_geo()

    def g(fips, k, default):
        rec = geo.get(fips)
        return rec[k] if rec and rec[k] is not None else default

    latest["county_name"] = [g(f, 0, f) for f in latest["county_fips"]]            # name or FIPS
    latest["state"] = [g(f, 1, f[:2]) for f in latest["county_fips"]]              # USPS or FIPS prefix
    latest["lat"] = [g(f, 2, STATE_CENTROID.get(f[:2], (39.5, -98.35))[0])
                     for f in latest["county_fips"]]
    latest["lon"] = [g(f, 3, STATE_CENTROID.get(f[:2], (39.5, -98.35))[1])
                     for f in latest["county_fips"]]
    latest["population"] = latest.get("total_population", 0)
    latest["poverty_rate"] = latest.get("poverty_rate", 0)

    cols = ["county_fips", "county_name", "state", "lat", "lon", "population", "poverty_rate",
            "risk", "ci_low", "ci_high", "shap1", "shap1_val", "shap2", "shap2_val",
            "shap3", "shap3_val", "confidence"]
    latest[cols].to_csv(OUT, index=False)
    print(f"Wrote {len(latest):,} real county predictions to {OUT}")


if __name__ == "__main__":
    main()
    