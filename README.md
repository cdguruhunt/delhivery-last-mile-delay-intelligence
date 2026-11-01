# Delhivery Last-Mile Delay Intelligence Pipeline

## Project Overview

This project develops an end-to-end data pipeline to analyse last-mile delivery performance, predict shipment delays, segment delivery hubs and recommend operational actions.

The project uses synthetic shipment data and is organised into four parts:

- **Part A:** Data preparation and exploratory data analysis.
- **Part B:** Delivery-time regression modelling.
- **Part C:** Late-delivery classification and alert threshold selection.
- **Part D:** Hub segmentation, model packaging and operational scoring.

All model results are based on synthetic data and must not be interpreted as actual Delhivery operational performance.

## How to run

### Requirements

Python dependencies are pinned in `requirements.txt`.

### Fresh-clone execution

Run the following commands in order:

```bash
git clone https://github.com/cdguruhunt/delhivery-last-mile-delay-intelligence.git
cd delhivery-last-mile-delay-intelligence
pip install -r requirements.txt

python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

The pipeline generates the synthetic source data, prepares the modelling table, executes the regression and classification experiments, creates hub segments, packages the final classifier and scores the new shipments.

`python package_model.py` rebuilds and saves `models/delay_model.pkl` using the final Part C configuration. It also creates the threshold, feature-order and imputation-value files required for prediction.

The prediction script reads `data/new_shipments.csv` and produces `outputs/new_shipment_actions.csv`.

### Model compatibility

The model is saved using joblib. Pickled scikit-learn models are version-sensitive. The committed model file should be loaded using the pinned dependency versions in `requirements.txt`.

If the installed scikit-learn version differs, run `python package_model.py` to regenerate the model in the current environment before running `python predict_delay.py`.

## Data findings

### Data quality and preparation

The synthetic historical dataset contained 1,224 rows, including 24 exact duplicate rows. After duplicate removal, 1,200 unique shipments remained.

The cleaning process standardised 12 inconsistent service-type spellings into three categories: Express, Standard and Economy.

Missing values were found in:

- Distance: 36
- Parcel weight: 24
- Declared value: 12
- Payment mode: 18

Five parcel-weight records contained scanner-fault sentinel values above 100 kg.

Missing distance, parcel weight and declared value were imputed using historical medians:

| Field | Imputation median |
| --- | --- |
| Distance | 314.95 km |
| Parcel weight | 1.83 kg |
| Declared value | INR 1,121.50 |

Missing payment modes were labelled `Unknown`.

The hub merge preserved all 1,200 shipment rows. The final modelling table contained no missing values across the 17 model features and the target.

### Delay findings

Overall, 278 of 1,200 shipments were delayed, representing a delay rate of 23.17%.

| Dimension | Category | Delay rate |
| --- | --- | --- |
| Service type | Express | 48.97% |
| Service type | Standard | 20.56% |
| Service type | Economy | 8.57% |
| Hub tier | Tier 1 | 4.00% |
| Hub tier | Tier 2 | 21.50% |
| Hub tier | Tier 3 | 44.00% |
| Distance | 0–200 km | 2.19% |
| Distance | 201–400 km | 13.68% |
| Distance | 401–600 km | 43.64% |
| Distance | Above 600 km | 85.23% |

### Correlation and declared-value transformation

The three features most strongly correlated with `delivery_hours`, excluding `delivery_hours` itself, were:

| Feature | Correlation |
| --- | --- |
| promised_hours | +0.8372 |
| distance_km | +0.4013 |
| n_stops | +0.3247 |

Declared value was right-skewed. Its mean was INR 1,533.0542 and its median was INR 1,121.50.

Applying the natural logarithm compressed the effect of large values. The transformed mean was 7.0277 and the median was 7.0224.

### Hub profile and segmentation findings

The hub profile contains 24 hubs, with 50 shipments per hub.

The following spot checks were verified:

| Hub | Mean delivery hours | Delay rate | Mean distance (km) | Mean stops |
| --- | --- | --- | --- | --- |
| HUB01 | 45.6850 | 2.00% | 185.061 | 54.70 |
| HUB13 | 66.0742 | 46.00% | 473.320 | 83.92 |
| HUB17 | 60.8514 | 46.00% | 443.285 | 84.18 |

K-Means was applied to four StandardScaler-transformed hub-level features: mean delivery hours, delay rate, mean distance and mean number of stops.

The separation evidence was:

| K | Across/within distance ratio | Cluster sizes |
| --- | --- | --- |
| 2 | 2.8940 | 15-9 |
| 3 | 5.3928 | 9-8-7 |
| 4 | 6.3047 | 8-7-5-4 |
| 5 | 6.8866 | 8-7-4-3-2 |

K=3 was retained as the operational segmentation. Although the separation ratio increases for K=4 and K=5, those alternatives produce clusters containing fewer than six hubs, making separate operational playbooks less practical.

The three segments were named in ascending order of mean hub-level delay rate:

| Segment | Hubs | Mean delivery hours | Mean delay rate | Mean distance (km) | Mean stops |
| --- | --- | --- | --- | --- | --- |
| Standard | 7 | 47.2396 | 2.57% | 188.9296 | 55.3686 |
| Watch | 8 | 56.8137 | 15.50% | 314.1762 | 70.7175 |
| Escalate | 9 | 64.6504 | 46.00% | 470.4757 | 84.3444 |

The tier-to-segment expectation is Tier 1 → Standard, Tier 2 → Watch and Tier 3 → Escalate.

Four hubs did not match this expectation:

| Hub | City | Tier | Actual segment | Expected segment |
| --- | --- | --- | --- | --- |
| HUB07 | Kolkata | 1 | Watch | Standard |
| HUB13 | Nagpur | 2 | Escalate | Watch |
| HUB15 | Kochi | 2 | Escalate | Watch |
| HUB22 | Dehradun | 3 | Watch | Escalate |

These differences should be reviewed rather than automatically treated as errors: the tier mapping and data-driven operational segment represent different classifications.

## Model results

### Regression — Part B

The modelling dataset contained 1,200 shipments and 17 input features. It was split into 840 training rows, 180 validation rows and 180 test rows using `random_state=42`.

StandardScaler was fitted only on training data. The maximum absolute mean of scaled training features was approximately 0.0000, compared with 0.2187 for the test features. This supports the intended separation of scaler fitting from test evaluation.

The mean baseline achieved a validation R² of -0.0212.

#### Model comparison

| Model | Validation MAE | Validation RMSE | Validation R² |
| --- | --- | --- | --- |
| Mean baseline | 15.7952 | 18.8552 | -0.0212 |
| Linear Regression | 3.9222 | 4.8912 | 0.9313 |
| Decision Tree | 5.0971 | 6.2796 | 0.8867 |
| Random Forest | 4.2882 | 5.5095 | 0.9128 |
| Gradient Boosting | 3.9166 | 4.9165 | 0.9306 |

Linear Regression achieved the lowest validation RMSE of 4.8912 hours. Gradient Boosting was a close second with RMSE of 4.9165 hours.

A depth-1 decision tree showed high bias, with training RMSE of 11.6334 and validation RMSE of 12.1324. The unlimited-depth tree showed high variance: training RMSE was 0.0000 while validation RMSE increased to 7.8368. A depth-6 tree achieved validation RMSE of 6.2796 hours.

A Gradient Boosting learning rate of 0.1 performed best in its experiment, with validation R² of 0.9306. The rates 0.01 and 1.0 achieved R² of 0.7127 and 0.8440 respectively.

#### Cross-validation and final model selection

Five-fold cross-validation on the combined training and validation data produced:

| Model | Mean R² | Standard deviation |
| --- | --- | --- |
| Linear Regression | 0.9261 | 0.0048 |
| Gradient Boosting | 0.9353 | 0.0045 |

Although Gradient Boosting achieved the stronger cross-validation mean, Linear Regression remained the final selection because it had the lowest validation RMSE, as required by the project brief.

#### Final regression test performance

| Metric | Test result |
| --- | --- |
| MAE | 3.5671 hours |
| MSE | 20.4624 |
| RMSE | 4.5235 hours |
| R² | 0.9493 |

The final test R² exceeds the project acceptance threshold of 0.88.

### Classification — Part C

The classification dataset was split using stratified sampling and `random_state=42`.

| Dataset | Rows | Delayed parcels | Delay rate |
| --- | --- | --- | --- |
| Training | 840 | 195 | 23.21% |
| Validation | 180 | 42 | 23.33% |
| Test | 180 | 41 | 22.78% |

#### Classification model comparison

The blind baseline predicts every parcel as on time.

| Model | TN | FP | FN | TP | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Blind all on time | 138 | 0 | 42 | 0 | 0.7667 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 128 | 10 | 7 | 35 | 0.9056 | 0.7778 | 0.8333 | 0.8046 |
| Decision Tree, depth 6 | 128 | 10 | 11 | 31 | 0.8833 | 0.7561 | 0.7381 | 0.7470 |
| Random Forest, default | 132 | 6 | 11 | 31 | 0.9056 | 0.8378 | 0.7381 | 0.7848 |

The blind baseline achieves 76.67% accuracy but has zero recall and zero F1 because it detects no delayed parcels. Accuracy alone is therefore insufficient to judge classifier usefulness.

#### Feature importance and selection

The top seven features ranked by ANOVA F-score were:

1. distance_km
2. hub_tier
3. hub_daily_capacity
4. n_stops
5. promised_hours
6. rain_flag
7. sale_event_flag

The five noise columns had a combined Random Forest importance of 0.2245. Their individual F-score ranks were all outside the top seven.

| Selected features | Validation F1 |
| --- | --- |
| 5 | 0.6988 |
| 7 | 0.8333 |
| 10 | 0.7857 |
| 12 | 0.8049 |
| 17 | 0.7848 |

Seven features produced the highest validation F1 of 0.8333.

#### Hyperparameter tuning

| Search method | Candidates | Total fits | Best CV accuracy | Validation F1 |
| --- | --- | --- | --- | --- |
| Grid Search | 27 | 135 | 0.8952 | 0.7750 |
| Randomized Search | 9 | 45 | 0.8905 | 0.7848 |
| Default forest, same 7 features | 1 | 1 | Not applicable | 0.8333 |

The default Random Forest using seven features achieved the highest validation F1. Hyperparameter tuning did not improve validation F1 in this experiment.

#### Operational alert threshold

The selected threshold was **0.5**, the highest tested threshold satisfying the validation recall requirement of at least 80%.

At this threshold:

- Validation precision: 0.7955
- Validation recall: 0.8333
- Validation F1: 0.8140

#### Final classification test performance

| Metric | Test result |
| --- | --- |
| Threshold | 0.5 |
| True negatives | 125 |
| False positives | 14 |
| False negatives | 10 |
| True positives | 31 |
| Accuracy | 0.8667 |
| Precision | 0.6889 |
| Recall | 0.7561 |
| F1 | 0.7209 |

Test recall was lower than validation recall. The test set contained only 41 delayed parcels, so each additional missed delayed parcel has a noticeable effect on recall.

#### Final classifier configuration

| Parameter | Selected value |
| --- | --- |
| Feature selector | SelectKBest |
| Selected features | 7 |
| Estimators | 100 |
| Maximum depth | Unlimited |
| Minimum samples per leaf | 1 |
| Alert threshold | 0.5 |
| Selection basis | default_forest_same_7_features |

## Operations recommendation

### Hub-level operating actions

Use the data-driven segment as an additional operational prioritisation layer:

- **Standard:** Maintain normal operating controls and monitor for deterioration.
- **Watch:** Review hub-level delay trends and investigate emerging operational issues.
- **Escalate:** Prioritise management attention, capacity and route reviews, and proactive exception handling.

The Escalate segment has an average hub-level delay rate of 46.00%. These hubs should receive the highest operational attention.

The four tier-versus-segment mismatches should be reviewed by the operations team. HUB07, HUB13 and HUB15 are in worse-performing segments than their tier mapping suggests. HUB22 is in a better-performing segment than its tier mapping suggests.

### Shipment-level alert actions

The classification threshold is 0.5. A shipment is flagged when its predicted probability of delay is greater than or equal to this threshold.

On the held-out test set:

- Precision of 0.6889 means approximately 6.9 out of every 10 flagged shipments were actually delayed.
- Recall of 0.7561 means approximately 7.6 out of every 10 delayed shipments were identified.

These are test-set results, not guaranteed production outcomes.

### New shipment scoring results

The scoring script was tested on 48 new synthetic shipments.

| Action | Shipments |
| --- | --- |
| No action | 40 |
| Proactive customer call | 2 |
| Escalate to hub manager | 6 |
| **Total** | **48** |

Eight of the 48 shipments were flagged (16.67%). Six required hub-manager escalation because they belonged to the Escalate segment. Two flagged shipments in other segments were assigned a proactive customer call.

The action rules are:

1. Flagged shipment in Escalate segment: `Escalate to hub manager`.
2. Flagged shipment in Standard or Watch segment: `Proactive customer call`.
3. Shipment not flagged: `No action`.

The 48-shipment exercise demonstrates the scoring workflow; it does not establish real-world alert effectiveness because the new shipment dataset has no observed delivery outcomes.

### Why three hub segments were retained

The separation ratio increases for K=4 and K=5. However, these alternatives produce clusters with fewer than six hubs. K=3 provides three operationally manageable groups of 9, 8 and 7 hubs, respectively, and was therefore retained for the operating playbook.

## Production plan

The following controls are proposed for a production implementation. They are recommendations, not results already demonstrated by the synthetic experiment.

### Production-readiness considerations

| Area | Proposed control | Repository evidence |
| --- | --- | --- |
| Accuracy | Track precision, recall, F1 and regression error against matured delivery outcomes. | `outputs/part_b_metrics.csv`, `outputs/part_c_final_test.csv` |
| Reliability | Validate required columns, missing values, hub joins and unique shipment IDs before scoring. | Shared preparation functions and prediction validation |
| Reproducibility | Pin dependencies, random seeds and model configuration. | `requirements.txt`, model scripts and final configuration |
| Performance | Measure batch scoring time and throughput as shipment volumes grow. | `predict_delay.py` |
| Maintainability | Keep shared cleaning and feature engineering in one module. | `src/shipment_prep.py` |
| Consistency | Use the same cleaning and feature-building functions for training and prediction, with stored imputation medians and feature order. | `package_model.py`, `predict_delay.py`, model artefacts |
| Monitorability | Record prediction volumes, alert rates, hub segments and matured outcome metrics. | `outputs/new_shipment_actions.csv` and scoring outputs |

### Proposed monitoring metrics and triggers

The following are suggested initial operating thresholds and should be calibrated with real production data.

1. **Hub delay rate:** Investigate a hub if its weekly delay rate rises by more than 5 percentage points compared with its trailing four-week average.
2. **Flagged shipment share:** Review the model and operational conditions if more than 25% of weekly scored shipments are flagged, or if the flagged share changes by more than 10 percentage points week over week.
3. **Matured-outcome recall:** Review the model if rolling four-week recall falls below 0.70 after delivery outcomes become available.
4. **Alert precision:** Review alert usefulness if rolling four-week precision falls below 0.60.

Monitoring should be performed at overall, hub, hub-segment and service-type levels.

### Retraining and drift monitoring

A quarterly model review and retraining cycle is proposed, with an additional review whenever monitoring thresholds are breached or material changes occur in operations.

Monitor changes in:

- Distance and number-of-stops distributions.
- Promised delivery hours and hub capacity.
- Service-type and hub-tier mix.
- Predicted delay probabilities and flagged shipment share.
- Actual delay rates and matured-outcome precision and recall.

The synthetic data includes a sale-event indicator. The sale window from September 15 to September 28 should be evaluated separately from ordinary periods.

For a production assessment, compare sale-event shipments with comparable non-sale periods, controlling where practical for service type, distance band and hub tier. Review delivery hours, delay rate, flagged share and matured-outcome precision and recall. A rise in delays during a sale window, or a change in feature distributions and prediction probabilities, may indicate operational or model drift.

Retraining should use newly matured, representative delivery outcomes and should be followed by validation before deployment.

## Limitations

1. **Synthetic data:** The dataset is generated and does not represent actual Delhivery shipment operations. Results cannot be treated as actual company performance.
2. **Limited test positives:** The classification test set contains only 41 delayed parcels. Precision and recall can change noticeably with a small number of additional errors.
3. **Threshold selection:** The alert threshold was selected using one validation split. Its performance may differ on future data.
4. **Small new-shipment sample:** The scoring demonstration contains only 48 synthetic shipments and has no observed outcome labels. It cannot establish production prediction accuracy.
5. **Hub segmentation sample:** Segmentation is based on 24 synthetic hub profiles. Cluster assignments and operational interpretations may change with real hub data.
6. **Model portability:** Joblib model files can be sensitive to Python and scikit-learn versions. The pinned environment and model regeneration procedure should be followed.
7. **Operational validation:** The proposed actions, monitoring triggers and retraining schedule require review and calibration with actual operations teams and production data.
