# Delhivery Last-Mile Delay Intelligence Pipeline

## Project Overview

This project develops a data-driven pipeline to analyse last-mile delivery
performance, predict delivery delays, and recommend operational actions
across delivery hubs.

The project uses synthetic shipment data and is organised into four parts:

- Part A: Data preparation and exploratory data analysis
- Part B: Delivery-time regression modelling
- Part C: Late-delivery classification
- Part D: Hub segmentation and operational recommendations

## How to Run

The project will be executable from a fresh clone of this repository.

Detailed execution instructions will be added after the pipeline scripts
and model have been developed and validated.

## Data Findings

### Data quality and preparation

The synthetic historical dataset contained 1,224 rows, including 24 exact
duplicate rows. After duplicate removal, 1,200 unique shipments remained.
The cleaning process standardised 12 inconsistent service-type spellings
into three categories: Express, Standard and Economy.

Missing values were found in distance (36), parcel weight (24), declared
value (12) and payment mode (18). Five parcel-weight records contained
scanner-fault sentinel values above 100 kg. Missing distance, parcel weight
and declared value were filled using their historical medians:
314.95 km, 1.83 kg and INR 1,121.50 respectively. Missing payment modes
were labelled Unknown.

The hub merge preserved all 1,200 shipment rows. The final modelling table
contained no missing values across the 17 model features and the target.

### Delay findings

Overall, 278 of 1,200 shipments were delayed, representing a delay rate
of 23.17%.

- **Service type:** Express had the highest delay rate at 48.97%, compared
  with Standard at 20.56% and Economy at 8.57%.
- **Hub tier:** Delay rates increased from 4.00% for Tier 1 to 21.50% for
  Tier 2 and 44.00% for Tier 3.
- **Distance:** Delay rates increased across distance bands: 2.19% for
  0–200 km, 13.68% for 201–400 km, 43.64% for 401–600 km and 85.23%
  for distances above 600 km.

### Correlation and declared-value transformation

The three features most strongly correlated with delivery_hours, excluding
delivery_hours itself, were:

1. promised_hours: +0.8372
2. distance_km: +0.4013
3. n_stops: +0.3247

Declared value was right-skewed: its mean was INR 1,533.0542 while its
median was INR 1,121.50. Applying the natural logarithm compressed the
effect of large values; the transformed mean was 7.0277 and the median
was 7.0224.

## Model Results

### Regression

To be completed after Part B.

### Classification

To be completed after Part C.

## Operations Recommendation

To be completed after Part D.

## Production Plan

To be completed after the end-to-end pipeline is validated.

## Limitations

- The project uses synthetic data, not actual Delhivery operational records.
- Model results are based on the generated dataset and should not be
  interpreted as actual company performance.


## Model results (regression)

### Data split and leakage prevention

The modelling dataset contained 1,200 shipments and 17 input features.
The data was split into 840 training rows, 180 validation rows and
180 test rows using random_state=42.

StandardScaler was fitted only on the training data. The maximum
absolute mean of the scaled training features was approximately
0.0000, while the corresponding test value was 0.2187, confirming
that the test data was not used to fit the scaler.

### Baseline performance

The mean baseline achieved a validation R² of -0.0212.
This indicates that predicting the training-set average does not
provide useful predictive performance compared with the regression
models.

### Model comparison

Linear Regression achieved the best validation RMSE of 4.8912 hours
and validation R² of 0.9313. Gradient Boosting was a close second,
with RMSE of 4.9165 hours and R² of 0.9306.

Random Forest achieved validation R² of 0.9128, while the
Decision Tree achieved 0.8867.

### Decision-tree bias and variance

A depth-1 tree showed high bias, with training RMSE of 11.6334
and validation RMSE of 12.1324.

The unlimited-depth tree showed high variance: its training RMSE
was 0.0000, while validation RMSE increased to 7.8368.

A depth-6 tree achieved the lowest validation RMSE of 6.2796 hours.

### Gradient Boosting learning-rate experiment

A learning rate of 0.1 performed best, achieving validation R²
of 0.9306.

A learning rate of 0.01 learned too slowly with 100 estimators,
achieving R² of 0.7127. A learning rate of 1.0 performed less
effectively, achieving R² of 0.8440, consistent with overly large
updates reducing generalisation performance.

### Cross-validation and final model selection

Five-fold cross-validation on the combined training and validation
data produced:

- Linear Regression: mean R² 0.9261, standard deviation 0.0048.
- Gradient Boosting: mean R² 0.9353, standard deviation 0.0045.

Although Gradient Boosting achieved the stronger cross-validation
mean, Linear Regression remained the final selection because it
had the lowest validation RMSE, as required by the project brief.

### Final test performance

The selected Linear Regression model was evaluated once on the
untouched test set.

- Test MAE: 3.5671 hours.
- Test MSE: 20.4624.
- Test RMSE: 4.5235 hours.
- Test R²: 0.9493.

The final test R² exceeds the project acceptance threshold of 0.88.
