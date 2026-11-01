import numpy as np
import pandas as pd


# Delivery promise in hours for each service type
PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96
}


# Exact 17 model input columns in the prescribed order
FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]


# Five bookkeeping columns
NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]


def clean_shipments(raw, medians=None):
    """
    Clean raw shipment data and fill missing values.

    Parameters
    ----------
    raw : pandas.DataFrame
        Raw shipment data.
    medians : dict or None
        Previously calculated median values. If None, calculate medians
        from the cleaned historical data.

    Returns
    -------
    tuple
        (clean_df, medians)
    """

    # Work on a copy so that the original DataFrame is not modified
    clean = raw.copy()

    # Step 1: Remove exact duplicate rows
    clean = clean.drop_duplicates().copy()

    # Step 2: Standardise service type
    clean["service_type"] = (
        clean["service_type"]
        .str.strip()
        .str.title()
    )

    # Step 3: Replace missing payment modes
    clean["payment_mode"] = clean["payment_mode"].fillna("Unknown")

    # Step 4: Convert declared value to numeric INR
    clean["declared_value_inr"] = pd.to_numeric(
        clean["declared_value"]
        .astype("string")
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False),
        errors="coerce"
    )

    # Remove the original text column
    clean = clean.drop(columns=["declared_value"])

    # Step 5: Convert dispatch_time to datetime
    clean["dispatch_time"] = pd.to_datetime(
        clean["dispatch_time"]
    )

    # Step 6: Treat scanner-fault weights as missing
    clean.loc[
        clean["parcel_weight_kg"] > 100,
        "parcel_weight_kg"
    ] = np.nan

    # Step 7: Calculate medians or reuse supplied historical medians
    median_columns = [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr"
    ]

    # Convert columns to float so decimal medians can be filled
    for column in median_columns:
        clean[column] = pd.to_numeric(
            clean[column], errors="coerce"
        ).astype(float)

    if medians is None:
        medians = {
            column: float(clean[column].median())
            for column in median_columns
        }
    else:
        medians = {
            column: float(medians[column])
            for column in median_columns
        }

    # Fill missing values using the median dictionary
    for column in median_columns:
        clean[column] = clean[column].fillna(medians[column])

    return clean, medians


def build_features(clean, hubs):
    """
    Attach hub information and create model input features.

    This function does not use delivery_hours, so it can also process
    shipments that have not yet been delivered.
    """

    # Work on a copy
    features_df = clean.copy()

    # Attach hub information through a left join
    hub_columns = [
        "hub_id",
        "hub_city",
        "region",
        "hub_tier",
        "hub_daily_capacity"
    ]

    features_df = features_df.merge(
        hubs[hub_columns],
        on="hub_id",
        how="left",
        validate="many_to_one"
    )

    # Add promised delivery hours
    features_df["promised_hours"] = (
        features_df["service_type"].map(PROMISED_HOURS)
    )

    # COD indicator: Unknown and Prepaid are both 0
    features_df["is_cod"] = (
        features_df["payment_mode"].eq("COD").astype(int)
    )

    # Weekend indicator: Saturday = 1, Sunday = 1
    features_df["is_weekend"] = (
        features_df["dispatch_time"].dt.dayofweek >= 5
    ).astype(int)

    # Extract dispatch hour
    features_df["dispatch_hour"] = (
        features_df["dispatch_time"].dt.hour
    )

    # Natural logarithm of declared value
    features_df["log_declared_value"] = np.log(
        features_df["declared_value_inr"]
    )

    return features_df
