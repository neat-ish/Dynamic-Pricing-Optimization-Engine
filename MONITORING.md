# Model Monitoring & Deployment Guide

**Project:** Dynamic-Pricing-Optimization-Engine
**Dataset:** Inside Airbnb -- listings.csv
**Champion Optimizer:** Bayesian -- uncertainty-aware pricing per listing

**Deployment context:** Weekly batch pricing pipeline -- per-listing price recommendation generation for short-term rental hosts

This pipeline consumes feature-engineered listing data, runs per-listing price optimization against the segment-specific Random Forest demand model, and produces recommended prices alongside confidence intervals (Bayesian) or point estimates (SLSQP) that feed into a downstream host-facing pricing surface or rule engine.

The pipeline does not serve real-time per-request pricing. Recommendations are written to a results table post-batch and consumed by the host interface on a weekly refresh cadence. Monitoring runs on a confirmed-outcome window with a ~3-week lag, reflecting the time required for booking and review data to accumulate and establish ground truth on demand response.

---

## 1. Introduction & Context

Short-term rental pricing is a revenue management problem at scale. Hosts who price statically leave yield on the table during high-demand periods and lose bookings during low-demand periods. This project builds a data-driven pricing engine on the Inside Airbnb dataset that learns the demand-price relationship from historical listing and review data, segments properties into three elasticity profiles (budget, mid-range, premium), and optimises per-listing prices subject to competitive positioning and elasticity constraints. The revenue impact is validated through a stratified A/B test with both statistical and practical significance checks. The pipeline is designed to integrate into a batch workflow where optimization runs weekly, prices are surfaced to hosts or a pricing rule engine, and model health is tracked against confirmed booking outcomes. All optimization and deployment decisions require the A/B test to clear four criteria -- statistical significance, positive direction, minimum practical threshold, and CI excluding zero -- before a ROLLOUT verdict is issued.

## 2. Business KPIs

### 2.1 Primary KPIs

| KPI | Definition | Target |
|---|---|---|
| Revenue lift vs. baseline | (Optimized revenue - Static revenue) / Static revenue | > 2% |
| Booking rate (demand proxy) | Reviews per month, normalized by segment | Stable or improving vs. prior 30 days |
| Price acceptance rate | % of recommended prices adopted (if host-facing) | > 60% |
| Average revenue per listing | Mean weekly revenue across active listings | Segment-specific baseline |

### 2.2 Operational KPIs

| KPI | Definition | Alert Threshold |
|---|---|---|
| Optimization convergence rate | % of listings where SLSQP converges successfully | < 95% |
| Bayesian CI width | Mean width of 95% CI on optimal price per segment | > 2x baseline (high demand uncertainty) |
| Price deviation from neighbourhood median | Mean absolute % deviation of recommended vs. local median | Alert if consistently > 35% |
| Segment distribution shift | % of listings in each segment over time | > 5% shift triggers re-segmentation review |
| Demand score PSI | PSI on predicted demand scores vs. training distribution | Alert > 0.10 / Retrain > 0.20 |

### 2.3 Monitoring Cadence

| Frequency | What to Check | Owner |
|---|---|---|
| Weekly | Revenue lift, booking rate, optimization convergence rate | Data Science |
| Monthly | Demand model RMSE on confirmed-outcome window, feature PSI | Data Science |
| Quarterly | Elasticity recalibration per segment, retraining decision | Data Science + Product |

---

## 3. Model Monitoring

### 3.1 Score Distribution Drift (PSI)

PSI is computed on predicted demand scores and on all input features monthly, comparing the current period against the training distribution.

| PSI | Interpretation | Action |
|---|---|---|
| < 0.10 | Stable | No action |
| 0.10 - 0.20 | Moderate shift | Monitor closely; cross-reference with feature importance |
| > 0.20 | Significant shift | Plan retrain if a high-importance feature is affected |

A PSI > 0.20 on price, `neighborhood_median_price`, or `review_scores_rating` (the top-3 demand drivers) is treated as a hard retrain trigger. PSI > 0.20 on lower-importance features is lower urgency.

### 3.2 Demand Model Performance Monitoring

Evaluate on a confirmed-outcome window -- listings where sufficient new review data has accumulated (typically 3 weeks post-recommendation) to measure actual demand response.

- RMSE on a rolling 3-week window -- alert if increase > 20% from deployment baseline
- Price coefficient stability per segment -- alert if elasticity estimate shifts > 15%
- Feature importance rank stability -- watch for top-3 features being displaced by previously low-importance features (signals population shift)

RMSE is the primary tracking metric. Elasticity stability is equally important because the optimizer's constraint bounds are derived from it -- an elasticity shift without a model retrain means the optimizer will apply the wrong bounds.

### 3.3 Optimization Output Monitoring

| Signal | How to Measure | Alert Condition |
|---|---|---|
| Price change distribution | Histogram of recommended % price changes per segment | Mean change > 30% in any segment (constraint may have been violated) |
| Revenue lift estimate | Median projected revenue lift per segment | Negative lift in any segment for 2+ consecutive weeks |
| SLSQP convergence | % of listings with successful optimizer exit | < 95% (indicates demand model instability or constraint infeasibility) |
| Bayesian CI width | Mean 95% CI width per segment | Width > 2x training-period baseline |

### 3.4 Retraining Criteria

Retrain the demand model when any two of the following are true simultaneously:

- Demand score PSI > 0.20 on price or neighbourhood median price
- RMSE increases > 20% from deployment baseline on a confirmed-outcome window
- Elasticity estimate shifts > 15% in any segment on fresh data
- A material population event occurs (new city expansion, platform pricing policy change)

---

## 4. Deployment Notes

This section is a lightweight handoff reference. The pipeline is batch-only by design.

### 4.1 Serving Pattern

- **Batch optimization:** Run per-listing price optimization weekly for all active listings. Write recommended prices to a results table.
- **Confidence intervals:** Bayesian optimization results include a 95% CI per listing. Listings where the CI is wide (high demand uncertainty) should be flagged for conservative pricing or manual review rather than automated application.
- **Real-time pricing:** Not supported by this pipeline. If sub-second per-request pricing is needed, a lightweight surrogate model trained on the optimization outputs should be served as a REST endpoint.

### 4.2 Artifacts for Versioning

Each deployment should store the following. All paths relative to the run directory.

- **Model artifacts (serving)**:
  demand_model_random_forest.pkl
  kmeans_k3.pkl
  segment_scaler.pkl
  demand_features.txt
  optimization_config.json

- **Evaluation artifacts (model quality)**:
  eval/demand_model_comparison.csv     -- RMSE/MAE/R2 per model per segment
  eval/feature_importance_seg{0,1,2}.csv
  eval/residual_plots_seg{0,1,2}.png
  eval/segmentation_report.csv

- **Optimization artifacts**:
  outputs/optimization_scipy_results.csv
  outputs/optimization_bayesian_results.csv
  outputs/method_comparison_summary.csv
  outputs/demand_revenue_curves_seg{0,1,2}.png

- **A/B test artifacts (deployment evidence)**:
  outputs/ab_test_summary.csv
  outputs/segment_ab_results.csv
  outputs/power_analysis.png
  outputs/revenue_distribution.png
  outputs/statistical_test_record.json

- **Data lineage**:
  data/data_snapshot_metadata.json
  data/missing_value_summary.csv
  data/feature_engineering_log.txt

- **Monitoring baselines**:
  baselines/training_feature_distributions.csv
  baselines/predicted_demand_distribution.csv
  baselines/deployment_kpis.csv

### 4.3 Dependency Pinning

The scoring environment must match training exactly.

| Package | Risk if unpinned |
|---|---|
| scikit-learn | Random Forest prediction and KMeans assignment can shift across minor versions |
| scipy | Optimizer convergence behaviour can differ across versions |
| pandas / numpy | Data type defaults and aggregation behaviour can change |

---

*Nitish Patnaik | github.com/neat-ish*
