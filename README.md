# NYC Yellow Taxi: EDA & Tip-Prediction Model Comparison (R)

[![R](https://img.shields.io/badge/R-4.4.0-276DC3?logo=r&logoColor=white)](https://www.r-project.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/mohammedthaqee/r-analysis)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end R analysis of NYC Yellow Taxi trip data (January 2023): exploratory data
analysis of fares, tips, and ride patterns, followed by a comparison of five regression models
plus an LSTM network for predicting tip amount. Originally developed on Kaggle.

> 📓 Originally written and run on Kaggle — see the [live Kaggle notebook](https://www.kaggle.com/code/mohammedthaqee/r-analysis).

## What's inside

**`notebooks/nyc-taxi-eda-and-models.ipynb`** — the full analysis, in two parts:

1. **EDA** — cleans a 10,000-row sample (down to ~9,369 rows after filtering) and explores
   trip distance, fare, and tip distributions; temporal patterns (hourly/daily/weekly); payment
   type and passenger count breakdowns; a pickup–dropoff heatmap; trip duration; tip % vs.
   total fare; revenue by hour; and top pickup/dropoff zones (joined against the TLC zone
   lookup table).
2. **Model comparison** — cleans and feature-engineers the *full* January 2023 file (fare,
   distance, passenger count, payment type, pickup hour, pickup day), does a 70/30 train/test
   split, and compares **Linear, Polynomial (degree 2), Ridge, Lasso, and Elastic Net**
   regression (via `glmnet`/`caret`/`recipes`) against an **LSTM network** (Keras) for
   predicting `tip_amount`, evaluated on MAE, RMSE, and R².

**`docs/`** — the written-up paper and supporting documentation for the model comparison study.

## Key EDA findings

- **Trip distance** is heavily right-skewed — ~85% of trips are under 5 miles, consistent with
  short-haul Manhattan grid travel; a thin tail extends past 20 miles (airport/inter-borough trips).
- **Fares** peak around \$8–12, with a secondary spike near \$60–70 (likely flat-rate airport fares).
- **~15% of trips report a \$0 tip**, correlated with cash payment; the remaining tips cluster
  roughly normally around 15–20% of the fare.
- **Rush-hour bimodality** — trip volume and revenue both peak 7–9 AM and 5–7 PM on weekdays,
  with flatter, later weekend patterns.
- **Manhattan dominates** pickup/dropoff density, with secondary hotspots at airports (JFK, LGA)
  and major transit hubs in Brooklyn/Queens.

## Model comparison

| Model | MAE ($) | RMSE ($) | R² |
|---|---|---|---|
| Linear | 1.44 | 2.51 | 0.558 |
| Polynomial (deg. 2) | 1.42 | 2.49 | 0.567 |
| Ridge (L2) | 1.44 | 2.52 | 0.558 |
| Lasso (L1) | 1.43 | 2.51 | 0.558 |
| Elastic Net | 1.43 | 2.51 | 0.558 |
| **LSTM** | **1.08** | **2.23** | **0.655** |

The five classical models cluster tightly (R² 0.558–0.567); Polynomial regression's small edge
suggests real but limited nonlinearity in the fare/distance → tip relationship. The LSTM —
trained with early stopping, dropout, and normalized inputs — captures temporal and nonlinear
tipping patterns (rush-hour effects, payment-type interactions, fare/tip heteroscedasticity)
that the classical models treat as independent features, improving MAE by ~25% over the best
classical model.

## Tech stack

- **R** 4.4.0 — `arrow`, `dplyr`, `tidyr`, `data.table`, `ggplot2`, `lubridate`, `skimr` (EDA)
- `caret`, `recipes`, `glmnet` (Ridge/Lasso/ElasticNet), `polyreg`, `keras` (LSTM), `Metrics`

## Running it locally

1. Install R (4.4+) and the packages above:
   ```r
   install.packages(c("arrow", "dplyr", "tidyr", "ggplot2", "skimr", "data.table",
                       "lubridate", "caret", "glmnet", "keras", "recipes", "Metrics", "tidyverse"))
   ```
   `keras` also requires a working Python/TensorFlow backend — see the
   [R keras install guide](https://tensorflow.rstudio.com/install/) if `library(keras)` fails.
2. Download the January 2023 Yellow Taxi trip data (`.parquet`) from the
   [NYC TLC trip record page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
   and update the file path in the notebook's data-loading cells.
3. Open `notebooks/nyc-taxi-eda-and-models.ipynb` in Jupyter with an R kernel (`IRkernel`),
   or run the equivalent code in RStudio.

## Repo structure

```
nyc-taxi-analysis/
├── notebooks/
│   └── nyc-taxi-eda-and-models.ipynb   # EDA + model comparison (incl. LSTM)
├── docs/
│   ├── NYC_Taxi_Demand_Study.pdf              # academic-style write-up
│   └── NYC_Taxi_Project_Documentation.docx    # detailed project documentation
├── README.md
└── LICENSE
```

## License

MIT — see [LICENSE](LICENSE).
