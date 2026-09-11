# Model card - laptop price estimator `v20260911T1241`

## What it does
Estimates the **asking price**, in Algerian dinars, of a used laptop listed on
Ouedkniss, from its specifications, condition, city and listing date.

It outputs a **range**, not a single number. That is deliberate: identical
configurations in this dataset sell between 2x and 9x apart, so a point estimate
claims precision the data does not contain.

## Intended use
- Helping a seller set an asking price, and a buyer sanity-check one.
- Ranking listings by how far below their estimate they are priced
  (the "deals" feed), *always* combined with the anomaly detector, because a deep
  discount is as likely to be a scam as a bargain.

## Out of scope
- **It does not predict sale price.** Every row of training data is an *asking*
  price. The gap between asking and final price is real, seller-specific, and
  entirely unmeasured here.
- Machines far outside the training distribution: servers, desktops, parts-only
  listings, and the premium tail (which is thin and where error is highest).
- Any use where being wrong costs money without a human in the loop.

## Training data
- Source: scraped Ouedkniss laptop listings, 13,004 used for fitting.
- Target: `price_corrected`, after troll-price removal and
  unit-convention correction.
- Held-out test rows: 3,251.

## Performance

| Split | R² | MAE (DZD) | Median APE |
|---|---|---|---|
| Stratified random (comparable to the original report) | 0.8394 | 18,017 | 10.3% |
| **Time-based (train <= 2024, test 2025) - the honest headline** | 0.8165 | 21,051 | 12.1% |
| Stratified, excluding unit-ambiguous rows | 0.8397 | 17,897 | 10.3% |

The time-based number is lower than the random-split number. That is expected and
is the point: listings span 2018-2025 and carry seven years of dinar inflation and
tech depreciation, so a random split lets the model interpolate within time
periods it has already seen.

### Baselines it has to beat

| Baseline | R² | MAE (DZD) | Median APE |
|---|---|---|---|
| baseline: global median [time_based] | -0.1069 | 63,033 | 42.1% |
| baseline: median of identical spec [time_based] | 0.0738 | 51,194 | 27.8% |
| baseline: component cost sum [time_based] | 0.0946 | 49,617 | 24.2% |


### Prediction interval
10th-90th percentile coverage: **72.4%**, median width 33,959 DZD (36% of price).

## Known limitations
- **Noise ceiling.** Identical spec rows sell 2-9x apart; predicting the perfect
  per-configuration mean would cap out near R² 0.95 in log space. Most of the
  remaining error is not in the spec columns at all - it is in listing text,
  seller reputation, photos, negotiability and urgency, none of which are scraped.
- **Geographic skew.** A large share of listings come from a handful of Algiers
  districts, so estimates for southern wilayas rest on far less data.
- **13,004 rows, far fewer unique configurations.** Effective
  sample size is closer to the number of distinct spec signatures than to the row count.
- **Condition is unstated for ~41% of listings.** The model learns a split for
  "missing" rather than guessing, but that is a real information gap.

## Reproducing
```
git checkout unknown
make build && make pipeline && make train
```
Trained 2026-09-11T12:41:10+00:00 with scikit-learn 1.5.2
on Python 3.11.16.

## Notes
Predicts ASKING price, not sale price. Identical specs in this dataset sell 2-9x apart, so the point estimate should be read together with the 10th-90th percentile range.
