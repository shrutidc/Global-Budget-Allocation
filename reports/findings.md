# Findings — Global Government Budgets, 1936–2026

A consolidated write-up of the analysis in `notebooks/`. All numbers are produced by the
committed notebooks and the `globalbudget` package; figures referenced live in
`reports/figures/`.

> **Data note.** Amounts are **nominal USD billions** (not inflation-adjusted). Category values
> are each function's *share* of that country-year's total budget, and the nine shares sum to 100.
> Treat absolute-dollar trends and forecasts as nominal.

## Dataset at a glance

- **45 countries**, years **1936–2026**, **3,654** country-year rows, 9 spending categories.
- **Zero missing values**, no duplicate country-years, percentage shares sum to 100 (±0.03 rounding).
- Coverage varies by country: USA/UK from 1936; most from 1946; China 1950, Germany 1949, South Korea 1948.

## 1. Who spends the most (latest year)

| Rank | Country | Total budget (USD bn) |
|-----:|---------|----------------------:|
| 1 | USA | ~13,133 |
| 2 | China | ~6,724 |
| 3 | Germany | ~2,602 |
| 4 | Japan | ~2,188 |
| 5 | France | ~1,972 |

Budgets are heavily right-skewed — a handful of economies dominate global public spending
(see `eda_budget_distribution.png`).

## 2. Where the money goes (median share across all country-years)

Education (13.2%), Infrastructure (13.0%) and Administration (12.9%) lead, followed by the
social bloc — Social Welfare (11.9%), State Transfers (11.4%), Health (9.6%). Defense (7.8%),
Agriculture (7.2%) and Interest Payments (5.3%) are typically the smallest slices
(`eda_category_boxplots.png`).

## 3. Guns vs butter

Across countries, the **average defense share fell from ~10.5% (1950) to ~7.3% (2020)**, while
combined **social spending rose from ~48% to ~51%** — a broad post-Cold-War "butter over guns"
shift (`policy_guns_vs_butter.png`).

The shift is uneven at the country level. Comparing each country's Cold-War-average defense share
to its 2022+ average:

- **Largest declines:** Saudi Arabia (−13.0 pp), Singapore (−11.0 pp), Pakistan (−10.6 pp).
- **Largest increases:** Russia (+10.0 pp), Poland (+3.4 pp), Israel (+3.1 pp) — consistent with
  recent regional security pressures (`policy_defense_change.png`).

## 4. Debt service crowds out social spending

Interest payments and social spending shares are **negatively correlated (r = −0.44)** across the
panel: country-years with heavier debt-service burdens tend to devote less of the budget to
education, health, welfare and transfers (`policy_interest_vs_social.png`).

## 5. Cross-country clusters

Clustering the 45 countries on their mean spending profiles (standardised, KMeans with k chosen by
silhouette) separates broadly into **defense-oriented**, **social-welfare-oriented**, and
**development/infrastructure-oriented** regimes. The PCA projection and per-cluster mean profiles
are in `03_cross_country.ipynb` (`cross_country_clusters.png`).

## 6. Forecasting

On total-budget series with the last 10 years held out, **exponential smoothing (ETS)** and
**ARIMA(1,1,1)** generally beat naïve and drift baselines. Accuracy degrades for economies with
structural breaks (currency-regime changes, rebasing), where MAPE rises. Forecasts are nominal-USD
trend extrapolations and should be read as scenarios (`forecast_backtest_india.png`,
`forecast_future_usa_china.png`).

## 7. Machine learning

Predicting a country-year's **Health budget share** from its other spending shares, year, log total
budget and country (time-based split, train ≤2010 / test >2010): the random forest tracks actuals
closely. Because category shares are compositional, much of the signal is the accounting constraint
plus country/era effects — a caveat, not a flaw, and a useful template for swapping in other targets
(`ml_pred_vs_actual.png`, `ml_feature_importance.png`).

## Caveats & next steps

- **Nominal, not real.** Adding a GDP or CPI deflator would enable real-terms and share-of-GDP
  analysis — the single highest-value extension.
- Category definitions are assumed consistent across countries; cross-country level comparisons of a
  single function should be read with that in mind.
- Natural extensions: probabilistic forecast intervals, `pmdarima`/Prophet auto-tuning, panel
  fixed-effects models, and per-capita normalisation if population data is joined in.
