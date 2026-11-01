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
