# Dynamic-Pricing-Optimization-Engine

End-to-end dynamic pricing pipeline for short-term rental listings on the Inside Airbnb dataset. Covering missing value analysis, exploratory analysis, feature engineering, property segmentation, demand modelling, price optimization using constrained gradient and Bayesian methods, and a revenue A/B test with statistical and practical significance evaluation. Champion optimizer: Bayesian (Gaussian Process) for uncertainty-aware pricing; SLSQP for bulk batch runs.

---

## Business Context

Short-term rental platforms operate in highly dynamic markets where a static pricing strategy leaves significant revenue on the table. Hosts who underprice relative to neighbourhood demand give up yield; hosts who overprice relative to their segment lose bookings entirely. This project builds an end-to-end dynamic pricing engine on the Inside Airbnb dataset to learn demand from historical listing data, segments properties by price-sensitivity profile, and optimises per-listing prices subject to competitive and elasticity constraints. The revenue impact is validated through a statistically rigorous A/B test. The pipeline is designed to be integrated into a weekly batch pricing workflow where per-listing price recommendations are generated, validated, and surfaced to hosts or a downstream pricing rule engine. Optimization and deployment decisions will be governed by a structured A/B test with both statistical and practical significance checks, ensuring pricing changes are data-driven and operationally justified.

## Architecture

```
+---------------------+
|    00_setup         |
|                     |
|  Create project     |
|  directories        |
|  Extract            |
|  listings.csv.gz    |
|  Audit price        |
|  missingness        |
+---------------------+
          |
          v
+---------------------+
|  01_missing_        |
|  prices             |
|                     |
|  Chi-square test    |
|  (categorical)      |
|  T-test (numeric)   |
|  Logistic AUC       |
|  (predict           |
|  missingness)       |
|  MCAR / MAR /       |
|  MNAR verdict       |
+---------------------+
          |
          v
+---------------------+
|    02_eda           |
|                     |
|  Price cleaning     |
|  ($, comma strip)   |
|  Outlier removal    |
|  (Isolation         |
|  Forest)            |
|  Log-price          |
|  validation         |
|  Room / property    |
|  type analysis      |
|  Neighbourhood      |
|  price variance     |
+---------------------+
          |
          v
+---------------------+
|  03_feature_        |
|  engineering        |
|                     |
|  Missing indicator  |
|  flags              |
|  % string parsing   |
|  Group-median       |
|  imputation         |
|  Price ratio        |
|  features           |
|  Host / review      |
|  features           |
|  Capacity features  |
+---------------------+
          |
          v
+---------------------+      +---------------------+
|  04_segmentation    |      |  models/            |
|                     |      |                     |
|  Elbow + Silhouette | ---> |  kmeans_k3.pkl      |
|  K-Means (k=3)      |      |  segment_scaler.pkl |
|  Segment profiles   |      |  demand_model_      |
|  Elasticity by      |      |  random_forest.pkl  |
|  segment            |      |  demand_features    |
+---------------------+      |  .txt               |
          |                  +---------------------+
          v
+---------------------+
|  05_demand_         |
|  modeling           |
|                     |
|  Target: reviews /  |
|  month (demand      |
|  proxy)             |
|  Baselines: Ridge,  |
|  MLP                |
|  Champion: Random   |
|  Forest             |
|  Price elasticity   |
|  per segment        |
|  70/15/15 split     |
+---------------------+
          |
          v
+---------------------+
|  06_optimization    |
|                     |
|  Approach 1:        |
|  SLSQP              |
|  (constrained       |
|  scipy)             |
|                     |
|  Approach 2:        |
|  Bayesian GP +      |
|  Expected           |
|  Improvement        |
|                     |
|  Constraints:       |
|  +/-30% of nbhd     |
|  median, elasticity |
|  bounds per segment |
+---------------------+
          |
          v
+---------------------+
|  07_ab_testing      |
|                     |
|  Control: current   |
|  (static) price     |
|  Treatment:         |
|  optimizer price    |
|                     |
|  Power analysis     |
|  (Monte Carlo)      |
|  Stratified         |
|  assignment by      |
|  segment            |
|                     |
|  Statistical:       |
|  - t-test revenue   |
|  - Bootstrap CI     |
|                     |
|  Practical:         |
|  - Cohen's d        |
|  - Revenue lift %   |
|                     |
|  Verdict:           |
|  ROLLOUT /          |
|  MONITOR /          |
|  DO_NOT_ROLLOUT     |
+---------------------+
```

---

## Dataset

**Source:** [Inside Airbnb](http://insideairbnb.com/get-the-data.html) -- listings.csv

| Property | Value |
|---|---|
| Listings | ~70,000+ (city-dependent) |
| Raw features | ~75 |
| Price missingness | ~10.8% |
| Demand proxy | Reviews per month |
| Segments | 3 (K-Means) |

---

## Notebook Summaries

### 00_setup.ipynb -- Environment and Data Setup

Creating the project directory structure (`data/raw`, `data/processed`, `models`, `outputs`) and decompressing `listings.csv.gz`. Validating column presence and surfaces the ~10.8% missingness rate in the `price` column -- the optimization target -- flagging it for pattern analysis in notebook 01 rather than silently dropping or imputing.

### 01_missing_prices.ipynb -- Missing Price Analysis

**Goal:** To determine whether price missingness is random (MCAR) or systematic (MAR/MNAR) before deciding how to handle null-price records.

Three tests are applied in sequence:

| Test | What it checks | Decision rule |
|---|---|---|
| Chi-square | Categorical feature values vs. missingness indicator | Reject MCAR if p < 0.05 |
| Independent t-test | Numeric feature means: missing vs. non-missing | Reject MCAR if p < 0.05 |
| Logistic regression AUC | Predict missingness from all features | MCAR if AUC near 0.50 |

**Decision:** Price nulls are not imputed. Imputing the optimization target would introduce fabricated signal directly into the revenue objective. All records with missing price are dropped after the missingness pattern is characterised. If missingness is confirmed MCAR, the drop is unbiased. If MAR/MNAR is detected, the limitation is documented.

### 02_eda.ipynb -- Exploratory Data Analysis

**Goal:** To Understand price structure and listing characteristics before feature engineering.

Key findings that directly shaped downstream decisions:

- **Price distribution:** Heavily right-skewed. Log-transformation validated as the appropriate modelling scale.
- **Outlier detection:** Isolation Forest removes extreme-value listings that would distort demand model training.
- **Room type premium:** Entire homes command a 2-3x median price premium over private/shared rooms -- justifying room type as a segmentation feature.
- **Neighbourhood variance:** Substantial price variation across neighbourhoods, validating `neighborhood_median_price` as an optimization constraint anchor.
- **Property type distribution:** Top 10 property types cover ~90% of listings. Long-tail types are grouped.

### 03_feature_engineering.ipynb -- Feature Engineering

**Goal:** Tp transform raw listing data into a modelling-ready feature set.

**Key decisions:**

- **Missing indicator flags:** Missingness in `bedrooms`, `bathrooms`, `host_response_rate`, `review_scores_*` is itself a signal. A missing response rate indicates the host has not opted into responsiveness tracking -- informative for demand.
- **Group-median imputation for review scores:** Review scores are imputed by `property_type` group median rather than global median. Properties of the same type have more similar quality profiles; global imputation suppresses the within-type signal that distinguishes budget from premium listings.
- **Percentage string parsing:** `host_response_rate` and `host_acceptance_rate` are stored as strings (`"95%"`). Converted to float before any modelling.

Features created:

| Category | Features |
|---|---|
| Price ratios | `price_per_person`, `price_per_bedroom`, `price_vs_neighborhood` |
| Missing flags | `bedrooms_missing`, `host_response_rate_missing`, `review_scores_rating_missing` |
| Host quality | `host_response_rate`, `host_acceptance_rate`, `host_is_superhost` |
| Review scores | `review_scores_rating`, `review_scores_cleanliness`, `review_scores_location` |
| Capacity | `beds_per_bedroom`, `bathrooms_per_bedroom`, `capacity_per_bedroom` |

### 04_segmentation.ipynb -- Property Segmentation

**Goal:** Partitioning listings into pricing segments with distinct demand and elasticity profiles so the optimizer can apply segment-appropriate constraints.

Optimal k selected via Elbow method (inertia) and Silhouette analysis. k=3 is chosen.

| Segment | Profile | Typical Price | Elasticity |
|---|---|---|---|
| 0 | Budget private/shared rooms, low review count | Low | High (price-sensitive) |
| 1 | Mid-range entire homes, good reviews | Mid | Moderate |
| 2 | Premium entire homes, high review scores, superhosts | High | Low (quality-driven) |

**Decision:** Segmentation is required before optimization because a single demand model fit to the full dataset would learn an average elasticity that misrepresents both budget and premium listings. Segment-specific elasticity bounds prevent the optimizer from recommending aggressive price increases for high-elasticity listings.

### 05_demand_modeling.ipynb -- Demand Modelling

**Goal:** To train a demand proxy model per segment to feed into the price optimizer.

**Target variable:** `reviews_per_month` -- a standard demand proxy in Airbnb literature where direct booking data is unavailable. More reviews per month signals higher booking frequency.

Models benchmarked: Ridge Regression (baseline), MLP, Random Forest.

**Why Random Forest as champion:** Linear models produce a constant price coefficient across all listing types, which is inappropriate for a nonlinear demand-price relationship. Random Forest captures nonlinear interactions between price, location, capacity, and review scores. MLP achieves similar accuracy but produces noisier elasticity estimates, which the optimizer relies on for constraint setting.

The price coefficient from Ridge provides an initial per-segment elasticity estimate. Random Forest feature importances confirm that price, neighbourhood median price, and review score are the top three demand drivers across all segments.

**Split:** 70/15/15 by listing index (no temporal dependency in cross-sectional listing data).

### 06_optimization.ipynb -- Price Optimization

**Goal:** Find the revenue-maximising price per listing subject to competitive and elasticity constraints.

Two methods are compared:

**Method 1: Constrained scipy (SLSQP)**
- Objective: maximizing `price * predicted_demand`
- Constraints: price within +/-30% of neighbourhood median; price change respects segment elasticity bounds
- Fast, deterministic, appropriate for bulk optimization across the full listing set

**Method 2: Bayesian Optimization (Gaussian Process + Expected Improvement)**
- Same objective and constraints as SLSQP
- Provides 95% CI on optimal price as a by-product of the GP posterior
- Sample-efficient: fewer objective evaluations needed to converge
- Preferred when the demand curve is uncertain (sparse review history, new listings)

**Decision to run both:** SLSQP produces point estimates cheaply for the full listing set. Bayesian optimization produces confidence intervals that are operationally useful -- a host facing high price uncertainty should act differently than one where the optimal price is tightly constrained. Where the two methods disagree substantially, it indicates high variance in the demand estimate for that listing.

Results are saved per-segment with median price change and revenue lift estimates, which feed directly into the A/B test.

### 07_ab_testing.ipynb -- Revenue A/B Test

**Goal:** To validate whether the optimized pricing strategy produces a statistically and practically significant revenue improvement over static pricing.

**Power analysis:** Monte Carlo simulation across sample sizes to determine required n for 80% power at alpha=0.05 and a 2% minimum detectable revenue lift.

**Group assignment:** Stratified randomization within each segment. This ensures treatment and control groups are balanced on segment composition -- preventing a confound where one group happens to contain more premium listings.

**Outcome simulation:** Demand response is simulated using elasticity estimates from notebook 05. Clearly documented as simulation -- in a live deployment, actual booking outcomes would replace the simulated demand.

**Statistical tests:**

| Test | What it measures |
|---|---|
| Independent t-test on revenue | Is the revenue difference larger than sampling noise? |
| 95% bootstrap CI | Uncertainty range on revenue delta without normality assumption |
| Cohen's d | Effect size -- is the difference large enough to act on? |

**Decision logic:**

```
Criteria evaluated (4 total):
  1. t-test on revenue:      p < 0.05
  2. Revenue lift:           > 0 (positive direction)
  3. Lift substantial:       > 2% (minimum practical threshold)
  4. Bootstrap CI:           excludes zero

n_criteria met:
  4/4  ->  ROLLOUT: pricing strategy is statistically and practically justified
  2-3  ->  MONITOR: directional signal but insufficient evidence; extend test
  0-1  ->  DO_NOT_ROLLOUT: treatment shows no meaningful improvement
```

Segment-level results are broken out separately. A strategy that works for premium listings may not be appropriate for budget listings, and a blanket rollout decision would mask heterogeneous effects.

---

## Key Design Decisions Summary

| Decision | Rationale |
|---|---|
| Price nulls dropped, not imputed | Imputing the optimization target introduces fabricated signal into the revenue objective |
| Group-median imputation for review scores | Within-type imputation preserves quality signal suppressed by global median |
| Random Forest demand model | Captures nonlinear price-demand relationship; constant elasticity from Ridge is inappropriate |
| Segmentation before optimization | A single elasticity estimate misrepresents budget and premium listings equally |
| Both SLSQP and Bayesian optimization | SLSQP for bulk runs; Bayesian for uncertainty quantification on individual listings |
| Stratified A/B assignment by segment | Prevents segment composition imbalance from confounding the revenue comparison |
| Demand proxy (reviews/month) | Direct booking data unavailable; review rate is the standard academic proxy |

---

## Results Summary

| Stage | Key Output |
|---|---|
| Missing price analysis | Missingness pattern characterised (MCAR/MAR); drop decision documented |
| Segmentation | 3 segments identified with distinct elasticity and price profiles |
| Demand model | Random Forest champion; price + neighbourhood median + reviews are top drivers |
| Optimization | Bayesian method preferred for uncertainty quantification; SLSQP for bulk runs |
| A/B test | Revenue lift validated with t-test and bootstrap CI; segment-level breakdown included |

---

## Project Structure

```
Dynamic-Pricing-Optimization-Engine/
|
|-- data/
|   |-- raw/                          # listings.csv (gitignored)
|   |-- processed/                    # Intermediate outputs (gitignored)
|
|-- models/
|   |-- kmeans_k3.pkl
|   |-- segment_scaler.pkl
|   |-- demand_model_random_forest.pkl
|   |-- demand_features.txt
|
|-- outputs/
|   |-- optimization_scipy_results.csv
|   |-- optimization_bayesian_results.csv
|   |-- ab_test_summary.csv
|
|-- 00_setup.ipynb
|-- 01_missing_prices.ipynb
|-- 02_eda.ipynb
|-- 03_feature_engineering.ipynb
|-- 04_segmentation.ipynb
|-- 05_demand_modeling.ipynb
|-- 06_optimization.ipynb
|-- 07_ab_testing.ipynb
|-- MONITORING.md
|-- requirements.txt
|-- .gitignore
|-- README.md
```

---

## Setup

```bash
git clone https://github.com/<your-username>/Dynamic-Pricing-Optimization-Engine.git
cd Dynamic-Pricing-Optimization-Engine
pip install -r requirements.txt
```

Place `listings.csv.gz` in `data/raw/` then run notebooks in order: 00 -> 01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07. Each notebook saves processed outputs to `data/processed/` for the next stage.

**requirements.txt**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
joblib
statsmodels
jupyter
```

---

## Monitoring

See `MONITORING.md` for the full monitoring guide: business KPIs, demand model drift detection, retraining criteria, and some light deployment notes.

---
*Nitish Patnaik | github.com/neat-ish*
