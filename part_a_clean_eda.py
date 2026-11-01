import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the repository root available for importing shared functions
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.shipment_prep import (
    clean_shipments,
    build_features,
    PROMISED_HOURS,
    FEATURES
)


def main():

    # ---------------------------------------------------------
    # 1. Load source datasets
    # ---------------------------------------------------------

    raw_path = PROJECT_ROOT / "data" / "raw_shipments.csv"
    hubs_path = PROJECT_ROOT / "data" / "hubs.csv"

    raw = pd.read_csv(raw_path)
    hubs = pd.read_csv(hubs_path)

    raw_rows = len(raw)

    # Keep a deduplicated copy for measuring pre-cleaning quality
    raw_dedup = raw.drop_duplicates().copy()

    duplicate_rows_removed = raw_rows - len(raw_dedup)

    # Capture missing-value counts before filling
    missing_distance_before = raw_dedup["distance_km"].isna().sum()
    missing_weight_before = raw_dedup["parcel_weight_kg"].isna().sum()
    missing_value_before = raw_dedup["declared_value"].isna().sum()
    missing_payment_before = raw_dedup["payment_mode"].isna().sum()

    service_spellings_before = raw_dedup["service_type"].nunique()

    sentinel_weight_rows = (
        raw_dedup["parcel_weight_kg"] > 100
    ).sum()

    # ---------------------------------------------------------
    # 2. Clean historical shipment data
    # ---------------------------------------------------------

    clean, medians = clean_shipments(raw)

    service_spellings_after = clean["service_type"].nunique()

    # ---------------------------------------------------------
    # 3. Build features and attach hub information
    # ---------------------------------------------------------

    modelling = build_features(clean, hubs)

    rows_after_hub_merge = len(modelling)

    # ---------------------------------------------------------
    # 4. Create the target variable
    # ---------------------------------------------------------

    modelling["is_delayed"] = (
        modelling["delivery_hours"] > modelling["promised_hours"]
    ).astype(int)

    # ---------------------------------------------------------
    # 5. Calculate data quality and distribution metrics
    # ---------------------------------------------------------

    q1 = modelling["delivery_hours"].quantile(0.25)
    q3 = modelling["delivery_hours"].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    hours_outlier_count = (
        (modelling["delivery_hours"] < lower_bound)
        | (modelling["delivery_hours"] > upper_bound)
    ).sum()

    declared_mean = modelling["declared_value_inr"].mean()
    declared_median = modelling["declared_value_inr"].median()

    log_value = np.log(modelling["declared_value_inr"])

    log_mean = log_value.mean()
    log_median = log_value.median()

    nulls_in_features_and_target = (
        modelling[FEATURES + ["delivery_hours", "is_delayed"]]
        .isna()
        .sum()
        .sum()
    )

    delayed_count = modelling["is_delayed"].sum()
    delay_rate = modelling["is_delayed"].mean()

    # ---------------------------------------------------------
    # 6. Create Part A checks output
    # ---------------------------------------------------------

    checks = {
        "raw_rows": raw_rows,
        "duplicate_rows_removed": duplicate_rows_removed,
        "rows_after_dedup": len(clean),
        "missing_distance_km_before": missing_distance_before,
        "missing_parcel_weight_before": missing_weight_before,
        "missing_declared_value_before": missing_value_before,
        "missing_payment_mode_before": missing_payment_before,
        "service_type_spellings_before": service_spellings_before,
        "service_type_spellings_after": service_spellings_after,
        "sentinel_weight_rows": sentinel_weight_rows,
        "median_distance_km": medians["distance_km"],
        "median_parcel_weight_kg": medians["parcel_weight_kg"],
        "median_declared_value_inr": medians["declared_value_inr"],
        "rows_after_hub_merge": rows_after_hub_merge,
        "nulls_in_features_and_target": nulls_in_features_and_target,
        "delayed_count": delayed_count,
        "delay_rate": delay_rate,
        "hours_q1": q1,
        "hours_q3": q3,
        "hours_lower_bound": lower_bound,
        "hours_upper_bound": upper_bound,
        "hours_outlier_count": hours_outlier_count,
        "declared_value_mean_inr": declared_mean,
        "declared_value_median_inr": declared_median,
        "log_declared_value_mean": log_mean,
        "log_declared_value_median": log_median
    }

    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    checks_df = pd.DataFrame(
        list(checks.items()),
        columns=["metric", "value"]
    )

    checks_df["value"] = pd.to_numeric(
        checks_df["value"], errors="coerce"
    ).round(4)

    checks_df.to_csv(
        outputs_dir / "part_a_checks.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 7. Delay evidence by service type, hub tier and distance
    # ---------------------------------------------------------

    group_results = []

    def add_group_results(data, dimension, group_column):

        grouped = data.groupby(
            group_column,
            observed=True,
            sort=True
        )

        for group_name, group_data in grouped:

            group_results.append({
                "dimension": dimension,
                "group": group_name,
                "shipments": len(group_data),
                "delayed": int(group_data["is_delayed"].sum()),
                "delay_rate": group_data["is_delayed"].mean(),
                "mean_hours": group_data["delivery_hours"].mean()
            })

    # Service type
    add_group_results(
        modelling,
        "service_type",
        "service_type"
    )

    # Hub tier
    add_group_results(
        modelling,
        "hub_tier",
        "hub_tier"
    )

    # Distance bands: right edge inclusive
    modelling["distance_band"] = pd.cut(
        modelling["distance_km"],
        bins=[0, 200, 400, 600, 2000],
        labels=["0-200", "201-400", "401-600", "601+"],
        right=True
    )

    add_group_results(
        modelling,
        "distance_band",
        "distance_band"
    )

    delay_groups = pd.DataFrame(group_results)

    for column in ["delay_rate", "mean_hours"]:
        delay_groups[column] = delay_groups[column].round(4)

    delay_groups.to_csv(
        outputs_dir / "part_a_delay_by_group.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 8. Correlation matrix
    # ---------------------------------------------------------

    numeric_columns = modelling.select_dtypes(
        include=np.number
    ).columns

    correlations = modelling[numeric_columns].corr()

    correlations = correlations.round(4)

    correlations.to_csv(
        outputs_dir / "part_a_correlations.csv",
        index=True,
        index_label="feature"
    )

    # ---------------------------------------------------------
    # 9. Save final modelling table
    # ---------------------------------------------------------

    modelling_path = PROJECT_ROOT / "data" / "modelling_table.csv"

    modelling.to_csv(
        modelling_path,
        index=False
    )

    # ---------------------------------------------------------
    # 10. Print execution summary
    # ---------------------------------------------------------

    print("PART A EXECUTION COMPLETED")
    print("--------------------------")

    print("Raw rows:", raw_rows)
    print("Duplicate rows removed:", duplicate_rows_removed)
    print("Clean rows:", len(clean))
    print("Rows after hub merge:", rows_after_hub_merge)
    print("Delayed shipments:", delayed_count)
    print("Delay rate:", round(delay_rate, 4))

    print("\\nCalculated medians:")
    for column, value in medians.items():
        print(column, ":", round(value, 4))

    print("\\nOutput files created:")
    print("data/modelling_table.csv")
    print("outputs/part_a_checks.csv")
    print("outputs/part_a_delay_by_group.csv")
    print("outputs/part_a_correlations.csv")


if __name__ == "__main__":
    main()
