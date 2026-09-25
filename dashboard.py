# """
# Disaster Prediction & Fairness-Aware Resource Allocation — Interactive Dashboard
# Q1 journal supplement (IJDRR). Single-file app: all real results are embedded below,
# so it runs with no data/ folder. If a data/ folder with matching files exists next to
# this script, those override the embedded copies (e.g. real per-county predictions from
# make_dashboard_data.py). No training at runtime.
# Run:  streamlit run dashboard.py
# """
# import io
# import json
# from pathlib import Path

# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import streamlit as st

# # ----------------------------------------------------------------------------- config
# NAVY, GREEN, RED, GRAY = "#003366", "#2E8B57", "#DC143C", "#666666"
# DATA = Path(__file__).parent / "data"

# st.set_page_config(page_title="Disaster Prediction & Fair Allocation",
#                    layout="wide")

# # ------------------------------------------------- embedded real results (no files needed)
# _METRICS = {
#   "prediction": {"auc": 0.796, "auc_ci": [0.789, 0.802], "pr_auc": 0.373,
#                  "pr_auc_ci": [0.360, 0.386], "brier": 0.086},
#   "spatial": {"mean": 0.820, "range": [0.705, 0.959], "n_states": 49},
#   "extreme": {"worst_1pct": 0.824, "worst_5pct": 0.818},
#   "fairness": {"gain_points": 42.5, "vuln_greedy": 27.5, "vuln_guardian": 70.0,
#                "abandoned_greedy": 1121},
#   "dynamic": {"from": 21.3, "to": 27.6, "relative_pct": 30},
#   "uncertainty": {"from": 24.9, "to": 34.0, "relative_pct": 37},
#   "conformal": {"target": 95, "empirical": 94.5, "unsure_pct": 15.9},
#   "delong": {"vs_historical_z": 35.6, "vs_nri_z": 51.4, "p": "<1e-12"},
#   "bias_robust": {"unbiased": 70.0, "biased_standard": 56.0, "biased_robust": 75.5},
#   "shap_top": [["County disaster history", 1.00], ["Property damage", 0.70],
#                ["Flood events", 0.50], ["Month (seasonality)", 0.35]],
#   "models": [["XGBoost (chosen)", 0.796, 0.373], ["Random Forest", 0.800, 0.350],
#              ["Logistic Regression", 0.772, 0.274], ["MLP (neural net)", 0.751, 0.295]],
# }
# _ALLOC = """supply,method,total,vulnerable,gini,abandoned
# 20,Greedy,20.0,11.0,0.994,1143
# 20,Max-min,20.0,20.0,0.000,0
# 20,Guardian,20.0,70.0,0.436,0
# 40,Greedy,40.0,27.5,0.975,1121
# 40,Proportional,40.0,52.7,0.090,0
# 40,Max-min,40.0,40.0,0.000,0
# 40,Guardian,40.0,70.0,0.152,0
# 60,Greedy,60.0,45.7,0.929,1067
# 60,Max-min,60.0,60.0,0.000,0
# 60,Guardian,60.0,70.0,0.036,0"""
# _CLIMATE = """scenario,budget_cut,total,vulnerable,greedy_vulnerable
# Current,0,40.0,70.0,27.5
# SSP2-4.5,0,39.4,70.0,27.5
# SSP5-8.5,0,39.0,70.0,27.5
# SSP5-8.5,25,29.3,70.0,19.0"""
# _FRONTIER = """floor,efficiency,equity,gini
# 0.00,70.3,18.4,0.816
# 0.10,66.8,54.7,0.453
# 0.20,61.9,76.1,0.239
# 0.30,54.2,90.4,0.096
# 0.40,40.0,100.0,0.000"""
# _COUNTIES = """county_fips,county_name,state,lat,lon,population,poverty_rate,risk,ci_low,ci_high,shap1,shap1_val,shap2,shap2_val,shap3,shap3_val,confidence
# 06037,Los Angeles,CA,34.05,-118.24,10014009,0.18,0.234,0.218,0.251,County disaster history,0.12,Property damage,0.08,Flood events,0.05,Medium
# 17031,Cook,IL,41.84,-87.68,5169517,0.14,0.121,0.103,0.139,County disaster history,0.10,Month (seasonality),0.04,Storm count,0.03,Low
# 48201,Harris,TX,29.86,-95.39,4713325,0.16,0.402,0.378,0.425,Property damage,0.14,Flood events,0.11,Tropical events,0.07,High
# 04013,Maricopa,AZ,33.35,-112.49,4420568,0.13,0.087,0.071,0.103,County disaster history,0.06,Storm count,0.03,Month (seasonality),0.02,Low
# 12086,Miami-Dade,FL,25.61,-80.50,2716940,0.17,0.451,0.420,0.482,Tropical events,0.16,Flood events,0.12,Property damage,0.09,High
# 06073,San Diego,CA,33.02,-116.77,3338330,0.12,0.156,0.138,0.174,County disaster history,0.08,Flood events,0.04,Property damage,0.03,Medium
# 36047,Kings,NY,40.64,-73.94,2559903,0.20,0.198,0.176,0.219,County disaster history,0.09,Property damage,0.06,Month (seasonality),0.04,Medium
# 53033,King,WA,47.49,-121.83,2252782,0.10,0.142,0.124,0.160,Flood events,0.07,County disaster history,0.05,Storm count,0.03,Medium
# 22071,Orleans,LA,30.07,-89.93,383997,0.23,0.428,0.401,0.456,Tropical events,0.15,Flood events,0.13,Property damage,0.10,High
# 48167,Galveston,TX,29.39,-94.90,342139,0.15,0.395,0.369,0.421,Tropical events,0.14,Flood events,0.11,Property damage,0.08,High
# 37129,New Hanover,NC,34.18,-77.87,225702,0.14,0.367,0.341,0.393,Tropical events,0.13,Flood events,0.10,Storm count,0.06,High
# 01097,Mobile,AL,30.68,-88.20,414809,0.18,0.341,0.316,0.366,Tropical events,0.12,Property damage,0.09,Flood events,0.07,High
# 28047,Harrison,MS,30.42,-89.08,208080,0.19,0.358,0.332,0.384,Tropical events,0.13,Flood events,0.10,Property damage,0.08,High
# 32003,Clark,NV,36.21,-115.01,2266715,0.13,0.069,0.054,0.084,County disaster history,0.05,Storm count,0.02,Month (seasonality),0.02,Low
# 26163,Wayne,MI,42.28,-83.28,1749343,0.21,0.134,0.116,0.152,County disaster history,0.07,Month (seasonality),0.04,Storm count,0.03,Low"""


# @st.cache_data
# def load():
#     def read_csv(name, embedded, **kw):
#         f = DATA / name
#         if f.exists():
#             return pd.read_csv(f, **kw)
#         return pd.read_csv(io.StringIO(embedded), **kw)

#     counties = read_csv("county_predictions.csv", _COUNTIES, dtype={"county_fips": str})
#     alloc = read_csv("allocation_results.csv", _ALLOC)
#     climate = read_csv("climate_stress.csv", _CLIMATE)
#     frontier = read_csv("frontier.csv", _FRONTIER)
#     mfile = DATA / "metrics_summary.json"
#     metrics = json.loads(mfile.read_text()) if mfile.exists() else _METRICS
#     return counties, alloc, climate, frontier, metrics


# counties, alloc, climate, frontier, M = load()

# st.title("Disaster Prediction & Fairness-Aware Resource Allocation")
# st.caption("Predicting US county federal-disaster declarations and allocating relief equitably. "
#            "Interactive supplement to the IJDRR submission.")

# tab0, tab1, tabmap, tab2, tab3, tab4, tab5 = st.tabs([
#     "Overview", "County Prediction", "Risk Map", "Fair Allocation",
#     "Climate Stress", "Results Summary", "Export & Share"])

# # ============================================================ TAB 0: Overview
# with tab0:
#     st.subheader("What this project does")

#     st.markdown(
#         "> **The problem.** When disasters strike, federal relief is limited — and the communities "
#         "least able to recover (poor, rural, under-resourced counties) often receive the least help, "
#         "because standard 'serve the biggest need first' planning favors large, populous areas.")

#     st.markdown(
#         "**This project is a two-stage system that fixes that.** First, a machine-learning model "
#         "(**XGBoost**) trained on **24 years of real US data** predicts, for every county each month, "
#         "the probability that storm activity will escalate into a **federal disaster declaration**. "
#         "Second, those predictions feed a fairness-aware allocation method we call **Guardian**, which "
#         "distributes limited relief while *mathematically guaranteeing* a minimum coverage level for "
#         "vulnerable counties — the ones a greedy approach would abandon. The output is a transparent, "
#         "validated tool for deciding **where to pre-position relief before disasters strike**, more "
#         "equitably.")

#     st.markdown("#### Key results at a glance")
#     k = st.columns(4)
#     k[0].metric("Prediction accuracy", "0.796 AUC", "beats baselines p<1e-12", delta_color="off")
#     k[1].metric("Vulnerable coverage", "27.5% → 70%", "+42.5 points", delta_color="off")
#     k[2].metric("Counties greedy abandons", "1,121/mo", "Guardian: 0", delta_color="off")
#     k[3].metric("Under climate + budget cut", "70% held", "equity guaranteed", delta_color="off")

#     st.markdown("#### How it works (pipeline)")
#     st.graphviz_chart("""
#     digraph G {
#       rankdir=LR; bgcolor="transparent";
#       node [shape=box style="rounded,filled" fontname="Arial" fontsize="11" margin="0.18,0.12"];
#       edge [color="#666666"];
#       d [label="1. Data gathered\\nNOAA storms\\nFEMA declarations\\nCensus demographics" fillcolor="#e8eef5"];
#       f [label="2. Feature table\\n327,783 county-months\\n43 features" fillcolor="#e8eef5"];
#       x [label="3. XGBoost model\\npredicts disaster risk\\nAUC 0.796" fillcolor="#cfe3d4"];
#       u [label="4. Uncertainty\\nensemble + conformal\\n95% coverage" fillcolor="#cfe3d4"];
#       g [label="5. Guardian allocation\\nequity floor for\\nvulnerable counties" fillcolor="#cfe3d4"];
#       o [label="6. Output\\nequitable relief\\npre-positioning plan" fillcolor="#f3e0e0"];
#       d -> f -> x -> u -> g -> o;
#     }
#     """)
#     st.caption("Stage 1 (blue to green): prediction.  Stage 2 (green to red): fair allocation.")

#     st.markdown("#### What each dashboard tab shows")
#     st.markdown(
#         "- **County Prediction** — for any county: disaster risk, confidence interval, the "
#         "**SHAP drivers** behind it, and a plain high/low-risk verdict.\n"
#         "- **Fair Allocation** — Guardian vs. greedy/proportional/max-min, and the "
#         "fairness–efficiency trade-off.\n"
#         "- **Climate Stress** — whether the equity guarantee survives IPCC warming scenarios.\n"
#         "- **Results Summary** — every headline number from the paper in one place.")

#     st.markdown("#### Contributions")
#     st.markdown(
#         "1. **Accurate, validated prediction** of county disaster escalation from real data "
#         "(AUC 0.796; 0.82 across unseen states; sharper on the worst months), significantly beating "
#         "a historical baseline and the federal NRI index (DeLong p < 1e-12).\n"
#         "2. **A fairness allocation method (Guardian) with a *proved* equity guarantee** — vulnerable "
#         "coverage rises from 27.5% to 70% (+42.5 pts), and the floor is mathematically guaranteed when "
#         "feasible, not just observed.\n"
#         "3. **Robustness shown on every axis reviewers probe** — across vulnerability definitions "
#         "(poverty/elderly/minority/SVI), across time, under forecast uncertainty (ensemble + conformal), "
#         "and against standard fairness baselines.\n"
#         "4. **Climate resilience with credible, IPCC-grounded scenarios** — Guardian holds 70% vulnerable "
#         "coverage even under SSP5-8.5 warming plus a budget cut.\n"
#         "5. **A bias-robustness fix** — when the model under-predicts vulnerable counties, subgroup "
#         "calibration restores their protection (56% → 75.5%).")

#     st.success(
#         "**In one line:** to our knowledge, the first disaster-allocation framework to pair accurate, "
#         "validated prediction with a *provable, robust* fairness guarantee — turning limited relief "
#         "into equitable, climate-resilient pre-positioning decisions.")

# # ============================================================ TAB 1: County prediction
# with tab1:
#     st.subheader("County Disaster Risk Prediction")
#     if (DATA / "county_predictions.csv").exists():
#         st.caption("Showing real model output (per-county risk, CIs, and SHAP from your trained model).")
#     else:
#         st.info("County-level values here are illustrative placeholders. Run "
#                 "`make_dashboard_data.py` and copy `county_predictions.csv` into `data/` to show "
#                 "real predictions.")
#     names = counties["county_name"] + ", " + counties["state"]
#     pick = st.selectbox("Select a county", names.tolist(), index=0)
#     row = counties.iloc[names.tolist().index(pick)]

#     left, right = st.columns([3, 2])
#     with left:
#         risk = float(row["risk"])
#         st.markdown(f"<h1 style='color:{NAVY};margin-bottom:0'>{risk*100:.1f}%</h1>",
#                     unsafe_allow_html=True)
#         st.markdown("chance of a federal disaster declaration")
#         st.caption(f"95% CI: [{row['ci_low']*100:.1f}%, {row['ci_high']*100:.1f}%]")

#         # plain-language verdict: is this county high or low risk?
#         if risk >= 0.40:
#             st.error("**Very high risk** — strong candidate for relief pre-positioning.")
#         elif risk >= 0.25:
#             st.warning("**Elevated risk** — monitor and consider pre-positioning.")
#         elif risk >= 0.15:
#             st.info("**Moderate risk** — keep on watch.")
#         else:
#             st.success("**Low risk** — unlikely to need federal relief this month.")

#         conf = row["confidence"]
#         st.markdown(f"**Model confidence:** {conf}  "
#                     f"<span style='color:{GRAY}'>(conformal; ~15.9% of county-months flagged "
#                     f"'uncertain' at 95%)</span>", unsafe_allow_html=True)
#     with right:
#         try:
#             import folium
#             from streamlit_folium import st_folium
#             color = RED if risk > 0.25 else GREEN
#             m = folium.Map(location=[row["lat"], row["lon"]], zoom_start=6,
#                            tiles="CartoDB positron")
#             folium.CircleMarker([row["lat"], row["lon"]], radius=12, color=color,
#                                 fill=True, fill_opacity=0.7,
#                                 popup=f"{pick}: {risk*100:.1f}%").add_to(m)
#             st_folium(m, height=300, width=None, returned_objects=[])
#         except Exception:
#             st.map(pd.DataFrame({"lat": [row["lat"]], "lon": [row["lon"]]}))

#     # SHAP: why this prediction? — as a clear horizontal bar chart
#     st.markdown("#### Why this prediction? — SHAP feature contributions")
#     sf = [f"{row[f'shap{i}']}" for i in (1, 2, 3)]
#     sv = [float(row[f"shap{i}_val"]) * 100 for i in (1, 2, 3)]
#     figs = go.Figure(go.Bar(x=sv[::-1], y=sf[::-1], orientation="h", marker_color=NAVY,
#                             text=[f"+{v:.0f}%" for v in sv[::-1]], textposition="outside"))
#     figs.update_layout(height=220, margin=dict(t=10, b=10),
#                        xaxis_title="contribution to risk (percentage points)")
#     st.plotly_chart(figs, width='stretch')
#     st.caption("SHAP shows how much each factor pushed this county's risk up. Bigger bar = bigger "
#                "driver of the prediction.")

#     exp = row[["county_name", "state", "risk", "ci_low", "ci_high",
#                "shap1", "shap2", "shap3"]].to_frame().T
#     st.download_button("Export this prediction (CSV)", exp.to_csv(index=False),
#                        f"prediction_{row['county_fips']}.csv", "text/csv")

# # ============================================================ MAP TAB: Risk spread
# @st.cache_data
# def _county_geojson():
#     import json
#     import urllib.request
#     url = ("https://raw.githubusercontent.com/plotly/datasets/master/"
#            "geojson-counties-fips.json")
#     with urllib.request.urlopen(url, timeout=25) as r:
#         return json.load(r)


# with tabmap:
#     st.subheader("Where disaster risk concentrates (US counties)")
#     st.markdown("Each county is shaded by its **predicted probability of a federal disaster "
#                 "declaration**. Red = higher risk, green = lower. This shows the *spread* of risk "
#                 "across the country.")
#     cdf = counties.copy()
#     cdf["fips"] = cdf["county_fips"].str.zfill(5)
#     cdf["risk_pct"] = (cdf["risk"] * 100).round(1)
#     if len(cdf) < 200:
#         st.info("Showing only the bundled sample counties. Run `make_dashboard_data.py` and copy "
#                 "`county_predictions.csv` into `data/` to map all ~3,200 counties.")

#     rendered = False
#     try:
#         gj = _county_geojson()
#         fig = px.choropleth(cdf, geojson=gj, locations="fips", color="risk_pct",
#                             color_continuous_scale="RdYlGn_r", scope="usa",
#                             hover_name="county_name",
#                             hover_data={"state": True, "risk_pct": True, "fips": False},
#                             labels={"risk_pct": "risk %"})
#         fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=540,
#                           coloraxis_colorbar_title="risk %")
#         st.plotly_chart(fig, width='stretch')
#         rendered = True
#     except Exception:
#         pass

#     if not rendered:  # offline fallback: dot map using county centroids (no boundaries needed)
#         fig = px.scatter_geo(cdf, lat="lat", lon="lon", color="risk_pct", scope="usa",
#                              color_continuous_scale="RdYlGn_r", hover_name="county_name",
#                              hover_data={"state": True, "risk_pct": True},
#                              labels={"risk_pct": "risk %"})
#         fig.update_traces(marker=dict(size=7, line=dict(width=0)))
#         fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=540,
#                           coloraxis_colorbar_title="risk %")
#         st.caption("County-boundary file unavailable offline — showing counties as points instead.")
#         st.plotly_chart(fig, width='stretch')

#     hi = cdf.nlargest(10, "risk")[["county_name", "state", "risk_pct"]]
#     st.markdown("**Highest-risk counties shown:**")
#     st.dataframe(hi.rename(columns={"county_name": "County", "state": "State",
#                                     "risk_pct": "Risk %"}), hide_index=True, width='stretch')

# # ============================================================ TAB 2: Fair allocation
# with tab2:
#     st.subheader("Fair Resource Allocation")
#     supply = st.slider("Relief supply (% of total predicted need)", 20, 60, 40, step=20)
#     sub = alloc[alloc["supply"] == supply].copy()

#     order = {"Guardian": 0, "Proportional": 1, "Max-min": 2, "Greedy": 3}
#     sub["o"] = sub["method"].map(order); sub = sub.sort_values("o").drop(columns="o")

#     def color_method(r):
#         c = {"Guardian": "#e6f4ea", "Greedy": "#fde8e8"}.get(r["method"], "")
#         return [f"background-color:{c}"] * len(r)

#     show = sub.rename(columns={"total": "Total %", "vulnerable": "Vulnerable %",
#                                "gini": "Gini", "abandoned": "Abandoned/mo"})[
#         ["method", "Total %", "Vulnerable %", "Gini", "Abandoned/mo"]]
#     st.dataframe(show.style.apply(color_method, axis=1)
#                  .format({"Total %": "{:.1f}", "Vulnerable %": "{:.1f}",
#                           "Gini": "{:.3f}", "Abandoned/mo": "{:.0f}"}),
#                  hide_index=True, width='stretch')

#     c1, c2 = st.columns(2)
#     with c1:
#         fig = go.Figure()
#         cmap = {"Guardian": GREEN, "Greedy": RED, "Proportional": NAVY, "Max-min": GRAY}
#         fig.add_bar(x=sub["method"], y=sub["vulnerable"],
#                     marker_color=[cmap.get(m, GRAY) for m in sub["method"]],
#                     text=[f"{v:.1f}%" for v in sub["vulnerable"]], textposition="outside")
#         fig.update_layout(title="Vulnerable-county coverage by method",
#                           yaxis_title="Vulnerable coverage (%)", yaxis_range=[0, 80],
#                           height=360, margin=dict(t=40))
#         st.plotly_chart(fig, width='stretch')
#     with c2:
#         g_ab = int(sub[sub["method"] == "Greedy"]["abandoned"].iloc[0]) if (sub["method"] == "Greedy").any() else 0
#         st.markdown(f"<div style='text-align:center'><span style='color:{GRAY}'>Counties "
#                     f"abandoned per month by Greedy</span><br>"
#                     f"<span style='font-size:64px;color:{RED};font-weight:700'>{g_ab:,}</span>"
#                     f"<br><span style='color:{GRAY}'>Guardian abandons 0</span></div>",
#                     unsafe_allow_html=True)

#     st.markdown("**Fairness–efficiency frontier** (sweet spot ≈ floor 0.2–0.3)")
#     ff = go.Figure()
#     ff.add_scatter(x=frontier["floor"], y=frontier["efficiency"], mode="lines+markers",
#                    name="Efficiency", line=dict(color=NAVY, width=3))
#     ff.add_scatter(x=frontier["floor"], y=frontier["equity"], mode="lines+markers",
#                    name="Equity (1−Gini)", line=dict(color=GREEN, width=3))
#     ff.add_vrect(x0=0.2, x1=0.3, fillcolor=GREEN, opacity=0.08, line_width=0)
#     ff.update_layout(xaxis_title="Equity floor", yaxis_title="Score (%)",
#                      height=360, margin=dict(t=20))
#     st.plotly_chart(ff, width='stretch')
#     st.download_button("Export allocation comparison (CSV)", sub.to_csv(index=False),
#                        f"allocation_supply{supply}.csv", "text/csv")

# # ============================================================ TAB 3: Climate stress
# with tab3:
#     st.subheader("Climate Stress Testing")
#     c1, c2 = st.columns(2)
#     scenario = c1.radio("Emissions scenario",
#                         ["Current", "SSP2-4.5", "SSP5-8.5"], horizontal=True)
#     cut = c2.select_slider("Budget cut (%)", options=[0, 25],
#                            value=25 if scenario == "SSP5-8.5" else 0)
#     sel = climate[(climate["scenario"] == scenario) & (climate["budget_cut"] == cut)]
#     if sel.empty:
#         sel = climate[climate["scenario"] == scenario].iloc[[0]]
#     rec = sel.iloc[0]

#     a, b = st.columns(2)
#     a.metric("Total coverage", f"{rec['total']:.1f}%",
#              delta=f"{rec['total']-40:.1f} vs current", delta_color="inverse")
#     b.metric("Vulnerable coverage", f"{rec['vulnerable']:.1f}%", delta="held",
#              delta_color="off")

#     fig = go.Figure()
#     fig.add_bar(x=["Guardian", "Greedy"], y=[rec["vulnerable"], rec["greedy_vulnerable"]],
#                 marker_color=[GREEN, RED],
#                 text=[f"{rec['vulnerable']:.0f}%", f"{rec['greedy_vulnerable']:.0f}%"],
#                 textposition="outside")
#     fig.update_layout(title=f"Vulnerable coverage — {scenario}"
#                             + (f" + {cut}% budget cut" if cut else ""),
#                       yaxis_title="Vulnerable coverage (%)", yaxis_range=[0, 80], height=380)
#     st.plotly_chart(fig, width='stretch')

#     if rec["greedy_vulnerable"] < 30:
#         st.warning(f"Under {scenario}{' + budget cut' if cut else ''}, the greedy baseline leaves "
#                    f"vulnerable coverage at ~{rec['greedy_vulnerable']:.0f}% and abandons ~1,121+ "
#                    f"counties/month, while Guardian holds {rec['vulnerable']:.0f}%.")
#     st.caption("Hazard factors derived from IPCC AR6 warming × ~7%/°C precipitation scaling "
#                "(SSP2-4.5: +19%; SSP5-8.5: +31%).")
#     st.download_button("Export climate results (CSV)", climate.to_csv(index=False),
#                        "climate_stress.csv", "text/csv")

# # ============================================================ TAB 4: Results summary
# with tab4:
#     st.subheader("Results Summary (paper metrics)")
#     p, s, e = M["prediction"], M["spatial"], M["extreme"]
#     r1 = st.columns(4)
#     r1[0].metric("ROC-AUC", f"{p['auc']:.3f}", f"CI [{p['auc_ci'][0]}, {p['auc_ci'][1]}]",
#                  delta_color="off")
#     r1[1].metric("PR-AUC", f"{p['pr_auc']:.3f}", delta_color="off")
#     r1[2].metric("Brier", f"{p['brier']:.3f}", delta_color="off")
#     r1[3].metric("Spatial AUC (LOSO)", f"{s['mean']:.3f}",
#                  f"{s['n_states']} states", delta_color="off")

#     r2 = st.columns(4)
#     r2[0].metric("Extreme months (worst 1%)", f"{e['worst_1pct']:.3f}", delta_color="off")
#     r2[1].metric("Fairness gain", f"+{M['fairness']['gain_points']:.1f} pts",
#                  f"{M['fairness']['vuln_greedy']:.1f}% → {M['fairness']['vuln_guardian']:.1f}%",
#                  delta_color="off")
#     r2[2].metric("Dynamic boost", f"+{M['dynamic']['relative_pct']}%",
#                  f"{M['dynamic']['from']}→{M['dynamic']['to']} worst-month", delta_color="off")
#     r2[3].metric("Uncertainty boost", f"+{M['uncertainty']['relative_pct']}%",
#                  f"{M['uncertainty']['from']}→{M['uncertainty']['to']} worst-case",
#                  delta_color="off")
#     st.success(f"XGBoost significantly beats baselines — DeLong {M['delong']['p']} "
#                f"(z={M['delong']['vs_historical_z']} vs historical, "
#                f"z={M['delong']['vs_nri_z']} vs NRI). Bias-robust Guardian restores vulnerable "
#                f"coverage under model bias ({M['bias_robust']['biased_standard']:.0f}% → "
#                f"{M['bias_robust']['biased_robust']:.0f}%).")

#     c1, c2 = st.columns(2)
#     with c1:
#         sh = pd.DataFrame(M["shap_top"], columns=["feature", "importance"])
#         fig = go.Figure(go.Bar(x=sh["importance"], y=sh["feature"], orientation="h",
#                                marker_color=NAVY))
#         fig.update_layout(title="Top SHAP features", height=320,
#                           yaxis=dict(autorange="reversed"), margin=dict(t=40))
#         st.plotly_chart(fig, width='stretch')
#     with c2:
#         md = pd.DataFrame(M["models"], columns=["Model", "AUC", "PR-AUC"])
#         st.markdown("**Model comparison**")
#         st.dataframe(md.style.format({"AUC": "{:.3f}", "PR-AUC": "{:.3f}"})
#                      .highlight_max(subset=["AUC"], color="#e6f4ea"),
#                      hide_index=True, width='stretch')
#     st.download_button("Export all metrics (JSON)", json.dumps(M, indent=2),
#                        "metrics_summary.json", "application/json")

# # ============================================================ TAB 5: Export & share
# with tab5:
#     st.subheader("Export & Share")
#     st.markdown("**Download results**")
#     d = st.columns(4)
#     d[0].download_button("County predictions (CSV)", counties.to_csv(index=False),
#                          "county_predictions.csv", "text/csv")
#     d[1].download_button("Allocation (CSV)", alloc.to_csv(index=False),
#                          "allocation_results.csv", "text/csv")
#     d[2].download_button("Climate (CSV)", climate.to_csv(index=False),
#                          "climate_stress.csv", "text/csv")
#     d[3].download_button("Metrics (JSON)", json.dumps(M, indent=2),
#                          "metrics_summary.json", "application/json")
#     st.caption("For 300-DPI figures, use each Plotly chart's camera icon (top-right on hover) "
#                "to export a publication-quality PNG.")



"""
Disaster Prediction & Fairness-Aware Resource Allocation — Interactive Dashboard
Q1 journal supplement (IJDRR). Single-file app: all real results are embedded below,
so it runs with no data/ folder. If a data/ folder with matching files exists next to
this script, those override the embedded copies (e.g. real per-county predictions from
make_dashboard_data.py). No training at runtime.
Run:  streamlit run dashboard.py
"""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------- config
NAVY, GREEN, RED, GRAY = "#003366", "#2E8B57", "#DC143C", "#666666"
DATA = Path(__file__).parent / "data"

st.set_page_config(page_title="Disaster Prediction & Fair Allocation",
                   layout="wide")

# ------------------------------------------------- embedded real results (no files needed)
_METRICS = {
  "prediction": {"auc": 0.796, "auc_ci": [0.789, 0.802], "pr_auc": 0.373,
                 "pr_auc_ci": [0.360, 0.386], "brier": 0.086},
  "spatial": {"mean": 0.820, "range": [0.705, 0.959], "n_states": 49},
  "extreme": {"worst_1pct": 0.824, "worst_5pct": 0.818},
  "fairness": {"gain_points": 42.5, "vuln_greedy": 27.5, "vuln_guardian": 70.0,
               "abandoned_greedy": 1121},
  "dynamic": {"from": 21.3, "to": 27.6, "relative_pct": 30},
  "uncertainty": {"from": 24.9, "to": 34.0, "relative_pct": 37},
  "conformal": {"target": 95, "empirical": 94.5, "unsure_pct": 15.9},
  "delong": {"vs_historical_z": 35.6, "vs_nri_z": 51.4, "p": "<1e-12"},
  "bias_robust": {"unbiased": 70.0, "biased_standard": 56.0, "biased_robust": 75.5},
  "shap_top": [["County disaster history", 1.00], ["Property damage", 0.70],
               ["Flood events", 0.50], ["Month (seasonality)", 0.35]],
  "models": [["XGBoost (chosen)", 0.796, 0.373], ["Random Forest", 0.800, 0.350],
             ["Logistic Regression", 0.772, 0.274], ["MLP (neural net)", 0.751, 0.295]],
}
_ALLOC = """supply,method,total,vulnerable,gini,abandoned
20,Greedy,20.0,11.0,0.994,1143
20,Max-min,20.0,20.0,0.000,0
20,Guardian,20.0,70.0,0.436,0
40,Greedy,40.0,27.5,0.975,1121
40,Proportional,40.0,52.7,0.090,0
40,Max-min,40.0,40.0,0.000,0
40,Guardian,40.0,70.0,0.152,0
60,Greedy,60.0,45.7,0.929,1067
60,Max-min,60.0,60.0,0.000,0
60,Guardian,60.0,70.0,0.036,0"""
_CLIMATE = """scenario,budget_cut,total,vulnerable,greedy_vulnerable
Current,0,40.0,70.0,27.5
SSP2-4.5,0,39.4,70.0,27.5
SSP5-8.5,0,39.0,70.0,27.5
SSP5-8.5,25,29.3,70.0,19.0"""
_FRONTIER = """floor,efficiency,equity,gini
0.00,70.3,18.4,0.816
0.10,66.8,54.7,0.453
0.20,61.9,76.1,0.239
0.30,54.2,90.4,0.096
0.40,40.0,100.0,0.000"""
_COUNTIES = """county_fips,county_name,state,lat,lon,population,poverty_rate,risk,ci_low,ci_high,shap1,shap1_val,shap2,shap2_val,shap3,shap3_val,confidence
06037,Los Angeles,CA,34.05,-118.24,10014009,0.18,0.234,0.218,0.251,County disaster history,0.12,Property damage,0.08,Flood events,0.05,Medium
17031,Cook,IL,41.84,-87.68,5169517,0.14,0.121,0.103,0.139,County disaster history,0.10,Month (seasonality),0.04,Storm count,0.03,Low
48201,Harris,TX,29.86,-95.39,4713325,0.16,0.402,0.378,0.425,Property damage,0.14,Flood events,0.11,Tropical events,0.07,High
04013,Maricopa,AZ,33.35,-112.49,4420568,0.13,0.087,0.071,0.103,County disaster history,0.06,Storm count,0.03,Month (seasonality),0.02,Low
12086,Miami-Dade,FL,25.61,-80.50,2716940,0.17,0.451,0.420,0.482,Tropical events,0.16,Flood events,0.12,Property damage,0.09,High
06073,San Diego,CA,33.02,-116.77,3338330,0.12,0.156,0.138,0.174,County disaster history,0.08,Flood events,0.04,Property damage,0.03,Medium
36047,Kings,NY,40.64,-73.94,2559903,0.20,0.198,0.176,0.219,County disaster history,0.09,Property damage,0.06,Month (seasonality),0.04,Medium
53033,King,WA,47.49,-121.83,2252782,0.10,0.142,0.124,0.160,Flood events,0.07,County disaster history,0.05,Storm count,0.03,Medium
22071,Orleans,LA,30.07,-89.93,383997,0.23,0.428,0.401,0.456,Tropical events,0.15,Flood events,0.13,Property damage,0.10,High
48167,Galveston,TX,29.39,-94.90,342139,0.15,0.395,0.369,0.421,Tropical events,0.14,Flood events,0.11,Property damage,0.08,High
37129,New Hanover,NC,34.18,-77.87,225702,0.14,0.367,0.341,0.393,Tropical events,0.13,Flood events,0.10,Storm count,0.06,High
01097,Mobile,AL,30.68,-88.20,414809,0.18,0.341,0.316,0.366,Tropical events,0.12,Property damage,0.09,Flood events,0.07,High
28047,Harrison,MS,30.42,-89.08,208080,0.19,0.358,0.332,0.384,Tropical events,0.13,Flood events,0.10,Property damage,0.08,High
32003,Clark,NV,36.21,-115.01,2266715,0.13,0.069,0.054,0.084,County disaster history,0.05,Storm count,0.02,Month (seasonality),0.02,Low
26163,Wayne,MI,42.28,-83.28,1749343,0.21,0.134,0.116,0.152,County disaster history,0.07,Month (seasonality),0.04,Storm count,0.03,Low"""


@st.cache_data
def load():
    def read_csv(name, embedded, **kw):
        f = DATA / name
        if f.exists():
            return pd.read_csv(f, **kw)
        return pd.read_csv(io.StringIO(embedded), **kw)

    counties = read_csv("county_predictions.csv", _COUNTIES, dtype={"county_fips": str})
    alloc = read_csv("allocation_results.csv", _ALLOC)
    climate = read_csv("climate_stress.csv", _CLIMATE)
    frontier = read_csv("frontier.csv", _FRONTIER)
    mfile = DATA / "metrics_summary.json"
    metrics = json.loads(mfile.read_text()) if mfile.exists() else _METRICS
    return counties, alloc, climate, frontier, metrics


counties, alloc, climate, frontier, M = load()

st.title("Disaster Prediction & Fairness-Aware Resource Allocation")
st.caption("Predicting US county federal-disaster declarations and allocating relief equitably. "
           "Interactive supplement to the IJDRR submission.")

tab0, tab1, tabmap, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "County Prediction", "Risk Map", "Fair Allocation",
    "Climate Stress", "Results Summary", "Export & Share"])

# ============================================================ TAB 0: Overview
with tab0:
    st.subheader("What this project does")

    st.markdown(
        "> **The problem.** When disasters strike, federal relief is limited — and the communities "
        "least able to recover (poor, rural, under-resourced counties) often receive the least help, "
        "because standard 'serve the biggest need first' planning favors large, populous areas.")

    st.markdown(
        "**This project is a two-stage system that fixes that.** First, a machine-learning model "
        "(**XGBoost**) trained on **24 years of real US data** predicts, for every county each month, "
        "the probability that storm activity will escalate into a **federal disaster declaration**. "
        "Second, those predictions feed a fairness-aware allocation method we call **Guardian**, which "
        "distributes limited relief while *mathematically guaranteeing* a minimum coverage level for "
        "vulnerable counties — the ones a greedy approach would abandon. The output is a transparent, "
        "validated tool for deciding **where to pre-position relief before disasters strike**, more "
        "equitably.")

    st.markdown("#### Key results at a glance")
    k = st.columns(4)
    k[0].metric("Prediction accuracy", "0.796 AUC", "beats baselines p<1e-12", delta_color="off")
    k[1].metric("Vulnerable coverage", "27.5% → 70%", "+42.5 points", delta_color="off")
    k[2].metric("Counties greedy abandons", "1,121/mo", "Guardian: 0", delta_color="off")
    k[3].metric("Under climate + budget cut", "70% held", "equity guaranteed", delta_color="off")

    st.markdown("#### How it works (pipeline)")
    st.graphviz_chart("""
    digraph G {
      rankdir=LR; bgcolor="transparent";
      node [shape=box style="rounded,filled" fontname="Arial" fontsize="11" margin="0.18,0.12"];
      edge [color="#666666"];
      d [label="1. Data gathered\\nNOAA storms\\nFEMA declarations\\nCensus demographics" fillcolor="#e8eef5"];
      f [label="2. Feature table\\n327,783 county-months\\n43 features" fillcolor="#e8eef5"];
      x [label="3. XGBoost model\\npredicts disaster risk\\nAUC 0.796" fillcolor="#cfe3d4"];
      u [label="4. Uncertainty\\nensemble + conformal\\n95% coverage" fillcolor="#cfe3d4"];
      g [label="5. Guardian allocation\\nequity floor for\\nvulnerable counties" fillcolor="#cfe3d4"];
      o [label="6. Output\\nequitable relief\\npre-positioning plan" fillcolor="#f3e0e0"];
      d -> f -> x -> u -> g -> o;
    }
    """)
    st.caption("Stage 1 (blue to green): prediction.  Stage 2 (green to red): fair allocation.")

    st.markdown("#### What each dashboard tab shows")
    st.markdown(
        "- **County Prediction** — for any county: disaster risk, confidence interval, the "
        "**SHAP drivers** behind it, and a plain high/low-risk verdict.\n"
        "- **Fair Allocation** — Guardian vs. greedy/proportional/max-min, and the "
        "fairness–efficiency trade-off.\n"
        "- **Climate Stress** — whether the equity guarantee survives IPCC warming scenarios.\n"
        "- **Results Summary** — every headline number from the paper in one place.")

    st.markdown("#### Contributions")
    st.markdown(
        "1. **Accurate, validated prediction** of county disaster escalation from real data "
        "(AUC 0.796; 0.82 across unseen states; sharper on the worst months), significantly beating "
        "a historical baseline and the federal NRI index (DeLong p < 1e-12).\n"
        "2. **A fairness allocation method (Guardian) with a *proved* equity guarantee** — vulnerable "
        "coverage rises from 27.5% to 70% (+42.5 pts), and the floor is mathematically guaranteed when "
        "feasible, not just observed.\n"
        "3. **Robustness shown on every axis reviewers probe** — across vulnerability definitions "
        "(poverty/elderly/minority/SVI), across time, under forecast uncertainty (ensemble + conformal), "
        "and against standard fairness baselines.\n"
        "4. **Climate resilience with credible, IPCC-grounded scenarios** — Guardian holds 70% vulnerable "
        "coverage even under SSP5-8.5 warming plus a budget cut.\n"
        "5. **A bias-robustness fix** — when the model under-predicts vulnerable counties, subgroup "
        "calibration restores their protection (56% → 75.5%).")

    st.success(
        "**In one line:** to our knowledge, the first disaster-allocation framework to pair accurate, "
        "validated prediction with a *provable, robust* fairness guarantee — turning limited relief "
        "into equitable, climate-resilient pre-positioning decisions.")

# ============================================================ TAB 1: County prediction
with tab1:
    st.subheader("County Disaster Risk Prediction")
    if (DATA / "county_predictions.csv").exists():
        st.caption("Showing real model output (per-county risk, CIs, and SHAP from your trained model).")
    else:
        st.info("County-level values here are illustrative placeholders. Run "
                "`make_dashboard_data.py` and copy `county_predictions.csv` into `data/` to show "
                "real predictions.")
    names = counties["county_name"] + ", " + counties["state"]
    pick = st.selectbox("Select a county", names.tolist(), index=0)
    row = counties.iloc[names.tolist().index(pick)]

    left, right = st.columns([3, 2])
    with left:
        risk = float(row["risk"])
        st.markdown(f"<h1 style='color:{NAVY};margin-bottom:0'>{risk*100:.1f}%</h1>",
                    unsafe_allow_html=True)
        st.markdown("chance of a federal disaster declaration")
        st.caption(f"95% CI: [{row['ci_low']*100:.1f}%, {row['ci_high']*100:.1f}%]")

        # plain-language verdict: is this county high or low risk?
        if risk >= 0.40:
            st.error("**Very high risk** — strong candidate for relief pre-positioning.")
        elif risk >= 0.25:
            st.warning("**Elevated risk** — monitor and consider pre-positioning.")
        elif risk >= 0.15:
            st.info("**Moderate risk** — keep on watch.")
        else:
            st.success("**Low risk** — unlikely to need federal relief this month.")

        conf = row["confidence"]
        st.markdown(f"**Model confidence:** {conf}  "
                    f"<span style='color:{GRAY}'>(conformal; ~15.9% of county-months flagged "
                    f"'uncertain' at 95%)</span>", unsafe_allow_html=True)
    with right:
        try:
            import folium
            from streamlit_folium import st_folium
            color = RED if risk > 0.25 else GREEN
            m = folium.Map(location=[row["lat"], row["lon"]], zoom_start=6,
                           tiles="CartoDB positron")
            folium.CircleMarker([row["lat"], row["lon"]], radius=12, color=color,
                                fill=True, fill_opacity=0.7,
                                popup=f"{pick}: {risk*100:.1f}%").add_to(m)
            st_folium(m, height=300, width=None, returned_objects=[])
        except Exception:
            st.map(pd.DataFrame({"lat": [row["lat"]], "lon": [row["lon"]]}))

    # SHAP: why this prediction? — as a clear horizontal bar chart
    st.markdown("#### Why this prediction? — SHAP feature contributions")
    sf = [f"{row[f'shap{i}']}" for i in (1, 2, 3)]
    sv = [float(row[f"shap{i}_val"]) * 100 for i in (1, 2, 3)]
    figs = go.Figure(go.Bar(x=sv[::-1], y=sf[::-1], orientation="h", marker_color=NAVY,
                            text=[f"+{v:.0f}%" for v in sv[::-1]], textposition="outside"))
    figs.update_layout(height=220, margin=dict(t=10, b=10),
                       xaxis_title="contribution to risk (percentage points)")
    st.plotly_chart(figs, width='stretch')
    st.caption("SHAP shows how much each factor pushed this county's risk up. Bigger bar = bigger "
               "driver of the prediction.")

    exp = row[["county_name", "state", "risk", "ci_low", "ci_high",
               "shap1", "shap2", "shap3"]].to_frame().T
    st.download_button("Export this prediction (CSV)", exp.to_csv(index=False),
                       f"prediction_{row['county_fips']}.csv", "text/csv")

# ============================================================ MAP TAB: Risk spread
@st.cache_data
def _county_geojson():
    import json
    import urllib.request
    url = ("https://raw.githubusercontent.com/plotly/datasets/master/"
           "geojson-counties-fips.json")
    with urllib.request.urlopen(url, timeout=25) as r:
        return json.load(r)


# Settings shared with step10_full_system.py
MAP_SCARCITY = 0.40        # supply = 40% of total need
MAP_VULN_PCT = 75          # vulnerable = poverty rate in top 25%
MAP_RISK_PCT = 75          # high risk  = predicted risk in top 25%
MAP_FLOORS = (0.70, 0.5, 0.4, 0.3, 0.2)


def _greedy(need, supply):
    alloc = np.zeros_like(need)
    remaining = float(supply)
    for i in np.argsort(-need):
        give = min(need[i], remaining)
        alloc[i] = give
        remaining -= give
        if remaining <= 1e-9:
            break
    return alloc


def _guardian(need, supply, vulnerable):
    """Same result as the Guardian LP in step10, solved directly: vulnerable counties
    get at least the floor (stepped down if supply can't cover it), and everyone is
    then lifted to the highest common coverage level r the supply allows."""
    fv = np.zeros_like(need)
    for fl in MAP_FLOORS:
        cand = np.where(vulnerable, fl, 0.0)
        if (cand * need).sum() <= supply:
            fv = cand
            break
    lo, hi = 0.0, 1.0
    for _ in range(60):                       # bisection on r
        r = (lo + hi) / 2
        if (np.maximum(fv, r) * need).sum() <= supply:
            lo = r
        else:
            hi = r
    return np.maximum(fv, lo) * need


@st.cache_data
def _map_table(counties):
    cdf = counties.copy()
    cdf["fips"] = cdf["county_fips"].astype(str).str.zfill(5)
    cdf["risk_pct"] = (cdf["risk"] * 100).round(1)

    risk_cut = np.nanpercentile(cdf["risk"], MAP_RISK_PCT)
    pov_cut = np.nanpercentile(cdf["poverty_rate"], MAP_VULN_PCT)
    hi_risk = cdf["risk"] >= risk_cut
    vuln = cdf["poverty_rate"] >= pov_cut
    cdf["category"] = np.select(
        [hi_risk & vuln, hi_risk & ~vuln, ~hi_risk & vuln],
        ["High risk + vulnerable", "High risk only", "Vulnerable only"],
        default="Lower risk")
    cdf.loc[cdf["poverty_rate"].isna(), "category"] = np.nan

    ok = cdf[["population", "risk", "poverty_rate"]].notna().all(axis=1)
    sub = cdf[ok].copy()
    need = (sub["risk"] * sub["population"]).to_numpy(float)
    keep = need > 0
    sub, need = sub[keep], need[keep]
    if len(sub):
        supply = MAP_SCARCITY * need.sum()
        vmask = (sub["poverty_rate"] >= pov_cut).to_numpy()
        sub["greedy_cov"] = (100 * _greedy(need, supply) / need).round(1)
        sub["guardian_cov"] = (100 * _guardian(need, supply, vmask) / need).round(1)
        sub["coverage_diff"] = (sub["guardian_cov"] - sub["greedy_cov"]).round(1)
        cdf = cdf.merge(sub[["fips", "greedy_cov", "guardian_cov", "coverage_diff"]],
                        on="fips", how="left")
    else:
        cdf["greedy_cov"] = cdf["guardian_cov"] = cdf["coverage_diff"] = np.nan
    return cdf


MAP_VIEWS = {
    "Predicted declaration risk": (
        "Each county is shaded by its **predicted probability of a federal disaster "
        "declaration**. Red = higher risk, green = lower."),
    "Risk × social vulnerability": (
        "Counties in the top 25% of predicted risk and/or the top 25% of poverty rate. "
        "**Dark red** counties are both high-risk and vulnerable: the places a purely "
        "damage-based allocation is most likely to underserve."),
    "Guardian vs greedy coverage": (
        "Difference in coverage (percentage points) between Guardian and greedy "
        "(damage-based) allocation at 40% supply. **Green** = Guardian gives this county "
        "more of its need; **red** = less. Greedy fully covers the largest-need counties "
        "and gives nothing to the rest; Guardian spreads supply and protects vulnerable counties."),
    "Prediction confidence": (
        "How certain the model is for each county, based on the width of the ensemble "
        "confidence interval (High = narrow interval)."),
}
CAT_COLORS = {"High risk + vulnerable": "#b2182b", "High risk only": "#ef8a62",
              "Vulnerable only": "#67a9cf", "Lower risk": "#d1e5f0"}
CONF_COLORS = {"High": "#2b7bba", "Medium": "#89bedc", "Low": "#dbe9f6"}


def _map_figure(cdf, view, gj):
    hover = {"state": True, "fips": False, "risk_pct": True}
    if gj is not None:
        common = dict(geojson=gj, locations="fips", scope="usa", hover_name="county_name")
        make = px.choropleth
    else:  # offline fallback: county centroids as points
        common = dict(lat="lat", lon="lon", scope="usa", hover_name="county_name")
        make = px.scatter_geo

    if view == "Predicted declaration risk":
        fig = make(cdf, color="risk_pct", color_continuous_scale="RdYlGn_r",
                   hover_data=hover, labels={"risk_pct": "risk %"}, **common)
    elif view == "Risk × social vulnerability":
        d = cdf.dropna(subset=["category"])
        fig = make(d, color="category", color_discrete_map=CAT_COLORS,
                   category_orders={"category": list(CAT_COLORS)},
                   hover_data={**hover, "poverty_rate": ":.1%"},
                   labels={"risk_pct": "risk %", "category": "County category",
                           "poverty_rate": "poverty rate"}, **common)
    elif view == "Guardian vs greedy coverage":
        d = cdf.dropna(subset=["coverage_diff"])
        lim = float(np.nanmax(np.abs(d["coverage_diff"]))) if len(d) else 1.0
        fig = make(d, color="coverage_diff", color_continuous_scale="RdYlGn",
                   range_color=(-lim, lim),
                   hover_data={**hover, "greedy_cov": True, "guardian_cov": True},
                   labels={"coverage_diff": "Guardian − greedy (pp)",
                           "greedy_cov": "greedy coverage %",
                           "guardian_cov": "Guardian coverage %",
                           "risk_pct": "risk %"}, **common)
    else:
        d = cdf.dropna(subset=["confidence"])
        fig = make(d, color="confidence", color_discrete_map=CONF_COLORS,
                   category_orders={"confidence": list(CONF_COLORS)},
                   hover_data={**hover, "ci_low": ":.3f", "ci_high": ":.3f"},
                   labels={"risk_pct": "risk %", "confidence": "Prediction confidence"},
                   **common)

    if gj is None:
        fig.update_traces(marker=dict(size=7, line=dict(width=0)))
    else:
        fig.update_traces(marker_line_width=0)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=540,
                      legend=dict(yanchor="bottom", y=0.02, xanchor="left", x=0.01))
    return fig


with tabmap:
    st.subheader("County-level views")
    view = st.radio("Map view", list(MAP_VIEWS), horizontal=True)
    st.markdown(MAP_VIEWS[view])

    cdf = _map_table(counties)
    if len(cdf) < 200:
        st.info("Showing only the bundled sample counties. Run `make_dashboard_data.py` and copy "
                "`county_predictions.csv` into `data/` to map all ~3,200 counties. (With the "
                "sample, the top-25% cutoffs and the allocation are computed on those few "
                "counties only.)")

    try:
        gj = _county_geojson()
    except Exception:
        gj = None
        st.caption("County-boundary file unavailable offline — showing counties as points instead.")
    st.plotly_chart(_map_figure(cdf, view, gj), width='stretch')

    if view == "Predicted declaration risk":
        hi = cdf.nlargest(10, "risk")[["county_name", "state", "risk_pct"]]
        st.markdown("**Highest-risk counties shown:**")
        st.dataframe(hi.rename(columns={"county_name": "County", "state": "State",
                                        "risk_pct": "Risk %"}),
                     hide_index=True, width='stretch')
    elif view == "Risk × social vulnerability":
        counts = cdf["category"].value_counts().reindex(list(CAT_COLORS)).fillna(0).astype(int)
        cols = st.columns(4)
        for c, (k, v) in zip(cols, counts.items()):
            c.metric(k, f"{v:,}")
    elif view == "Guardian vs greedy coverage":
        d = cdf.dropna(subset=["coverage_diff"])
        cols = st.columns(3)
        cols[0].metric("Counties gaining under Guardian", f"{int((d['coverage_diff'] > 0.5).sum()):,}")
        cols[1].metric("Counties losing under Guardian", f"{int((d['coverage_diff'] < -0.5).sum()):,}")
        cols[2].metric("Counties greedy gives nothing", f"{int((d['greedy_cov'] <= 0).sum()):,}")
    else:
        counts = cdf["confidence"].value_counts().reindex(list(CONF_COLORS)).fillna(0).astype(int)
        cols = st.columns(3)
        for c, (k, v) in zip(cols, counts.items()):
            c.metric(f"{k} confidence", f"{v:,}")

# ============================================================ TAB 2: Fair allocation
with tab2:
    st.subheader("Fair Resource Allocation")
    supply = st.slider("Relief supply (% of total predicted need)", 20, 60, 40, step=20)
    sub = alloc[alloc["supply"] == supply].copy()

    order = {"Guardian": 0, "Proportional": 1, "Max-min": 2, "Greedy": 3}
    sub["o"] = sub["method"].map(order); sub = sub.sort_values("o").drop(columns="o")

    def color_method(r):
        c = {"Guardian": "#e6f4ea", "Greedy": "#fde8e8"}.get(r["method"], "")
        return [f"background-color:{c}"] * len(r)

    show = sub.rename(columns={"total": "Total %", "vulnerable": "Vulnerable %",
                               "gini": "Gini", "abandoned": "Abandoned/mo"})[
        ["method", "Total %", "Vulnerable %", "Gini", "Abandoned/mo"]]
    st.dataframe(show.style.apply(color_method, axis=1)
                 .format({"Total %": "{:.1f}", "Vulnerable %": "{:.1f}",
                          "Gini": "{:.3f}", "Abandoned/mo": "{:.0f}"}),
                 hide_index=True, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        cmap = {"Guardian": GREEN, "Greedy": RED, "Proportional": NAVY, "Max-min": GRAY}
        fig.add_bar(x=sub["method"], y=sub["vulnerable"],
                    marker_color=[cmap.get(m, GRAY) for m in sub["method"]],
                    text=[f"{v:.1f}%" for v in sub["vulnerable"]], textposition="outside")
        fig.update_layout(title="Vulnerable-county coverage by method",
                          yaxis_title="Vulnerable coverage (%)", yaxis_range=[0, 80],
                          height=360, margin=dict(t=40))
        st.plotly_chart(fig, width='stretch')
    with c2:
        g_ab = int(sub[sub["method"] == "Greedy"]["abandoned"].iloc[0]) if (sub["method"] == "Greedy").any() else 0
        st.markdown(f"<div style='text-align:center'><span style='color:{GRAY}'>Counties "
                    f"abandoned per month by Greedy</span><br>"
                    f"<span style='font-size:64px;color:{RED};font-weight:700'>{g_ab:,}</span>"
                    f"<br><span style='color:{GRAY}'>Guardian abandons 0</span></div>",
                    unsafe_allow_html=True)

    st.markdown("**Fairness–efficiency frontier** (sweet spot ≈ floor 0.2–0.3)")
    ff = go.Figure()
    ff.add_scatter(x=frontier["floor"], y=frontier["efficiency"], mode="lines+markers",
                   name="Efficiency", line=dict(color=NAVY, width=3))
    ff.add_scatter(x=frontier["floor"], y=frontier["equity"], mode="lines+markers",
                   name="Equity (1−Gini)", line=dict(color=GREEN, width=3))
    ff.add_vrect(x0=0.2, x1=0.3, fillcolor=GREEN, opacity=0.08, line_width=0)
    ff.update_layout(xaxis_title="Equity floor", yaxis_title="Score (%)",
                     height=360, margin=dict(t=20))
    st.plotly_chart(ff, width='stretch')
    st.download_button("Export allocation comparison (CSV)", sub.to_csv(index=False),
                       f"allocation_supply{supply}.csv", "text/csv")

# ============================================================ TAB 3: Climate stress
with tab3:
    st.subheader("Climate Stress Testing")
    c1, c2 = st.columns(2)
    scenario = c1.radio("Emissions scenario",
                        ["Current", "SSP2-4.5", "SSP5-8.5"], horizontal=True)
    cut = c2.select_slider("Budget cut (%)", options=[0, 25],
                           value=25 if scenario == "SSP5-8.5" else 0)
    sel = climate[(climate["scenario"] == scenario) & (climate["budget_cut"] == cut)]
    if sel.empty:
        sel = climate[climate["scenario"] == scenario].iloc[[0]]
    rec = sel.iloc[0]

    a, b = st.columns(2)
    a.metric("Total coverage", f"{rec['total']:.1f}%",
             delta=f"{rec['total']-40:.1f} vs current", delta_color="inverse")
    b.metric("Vulnerable coverage", f"{rec['vulnerable']:.1f}%", delta="held",
             delta_color="off")

    fig = go.Figure()
    fig.add_bar(x=["Guardian", "Greedy"], y=[rec["vulnerable"], rec["greedy_vulnerable"]],
                marker_color=[GREEN, RED],
                text=[f"{rec['vulnerable']:.0f}%", f"{rec['greedy_vulnerable']:.0f}%"],
                textposition="outside")
    fig.update_layout(title=f"Vulnerable coverage — {scenario}"
                            + (f" + {cut}% budget cut" if cut else ""),
                      yaxis_title="Vulnerable coverage (%)", yaxis_range=[0, 80], height=380)
    st.plotly_chart(fig, width='stretch')

    if rec["greedy_vulnerable"] < 30:
        st.warning(f"Under {scenario}{' + budget cut' if cut else ''}, the greedy baseline leaves "
                   f"vulnerable coverage at ~{rec['greedy_vulnerable']:.0f}% and abandons ~1,121+ "
                   f"counties/month, while Guardian holds {rec['vulnerable']:.0f}%.")
    st.caption("Hazard factors derived from IPCC AR6 warming × ~7%/°C precipitation scaling "
               "(SSP2-4.5: +19%; SSP5-8.5: +31%).")
    st.download_button("Export climate results (CSV)", climate.to_csv(index=False),
                       "climate_stress.csv", "text/csv")

# ============================================================ TAB 4: Results summary
with tab4:
    st.subheader("Results Summary (paper metrics)")
    p, s, e = M["prediction"], M["spatial"], M["extreme"]
    r1 = st.columns(4)
    r1[0].metric("ROC-AUC", f"{p['auc']:.3f}", f"CI [{p['auc_ci'][0]}, {p['auc_ci'][1]}]",
                 delta_color="off")
    r1[1].metric("PR-AUC", f"{p['pr_auc']:.3f}", delta_color="off")
    r1[2].metric("Brier", f"{p['brier']:.3f}", delta_color="off")
    r1[3].metric("Spatial AUC (LOSO)", f"{s['mean']:.3f}",
                 f"{s['n_states']} states", delta_color="off")

    r2 = st.columns(4)
    r2[0].metric("Extreme months (worst 1%)", f"{e['worst_1pct']:.3f}", delta_color="off")
    r2[1].metric("Fairness gain", f"+{M['fairness']['gain_points']:.1f} pts",
                 f"{M['fairness']['vuln_greedy']:.1f}% → {M['fairness']['vuln_guardian']:.1f}%",
                 delta_color="off")
    r2[2].metric("Dynamic boost", f"+{M['dynamic']['relative_pct']}%",
                 f"{M['dynamic']['from']}→{M['dynamic']['to']} worst-month", delta_color="off")
    r2[3].metric("Uncertainty boost", f"+{M['uncertainty']['relative_pct']}%",
                 f"{M['uncertainty']['from']}→{M['uncertainty']['to']} worst-case",
                 delta_color="off")
    st.success(f"XGBoost significantly beats baselines — DeLong {M['delong']['p']} "
               f"(z={M['delong']['vs_historical_z']} vs historical, "
               f"z={M['delong']['vs_nri_z']} vs NRI). Bias-robust Guardian restores vulnerable "
               f"coverage under model bias ({M['bias_robust']['biased_standard']:.0f}% → "
               f"{M['bias_robust']['biased_robust']:.0f}%).")

    c1, c2 = st.columns(2)
    with c1:
        sh = pd.DataFrame(M["shap_top"], columns=["feature", "importance"])
        fig = go.Figure(go.Bar(x=sh["importance"], y=sh["feature"], orientation="h",
                               marker_color=NAVY))
        fig.update_layout(title="Top SHAP features", height=320,
                          yaxis=dict(autorange="reversed"), margin=dict(t=40))
        st.plotly_chart(fig, width='stretch')
    with c2:
        md = pd.DataFrame(M["models"], columns=["Model", "AUC", "PR-AUC"])
        st.markdown("**Model comparison**")
        st.dataframe(md.style.format({"AUC": "{:.3f}", "PR-AUC": "{:.3f}"})
                     .highlight_max(subset=["AUC"], color="#e6f4ea"),
                     hide_index=True, width='stretch')
    st.download_button("Export all metrics (JSON)", json.dumps(M, indent=2),
                       "metrics_summary.json", "application/json")

# ============================================================ TAB 5: Export & share
with tab5:
    st.subheader("Export & Share")
    st.markdown("**Download results**")
    d = st.columns(4)
    d[0].download_button("County predictions (CSV)", counties.to_csv(index=False),
                         "county_predictions.csv", "text/csv")
    d[1].download_button("Allocation (CSV)", alloc.to_csv(index=False),
                         "allocation_results.csv", "text/csv")
    d[2].download_button("Climate (CSV)", climate.to_csv(index=False),
                         "climate_stress.csv", "text/csv")
    d[3].download_button("Metrics (JSON)", json.dumps(M, indent=2),
                         "metrics_summary.json", "application/json")
    st.caption("For 300-DPI figures, use each Plotly chart's camera icon (top-right on hover) "
               "to export a publication-quality PNG.")