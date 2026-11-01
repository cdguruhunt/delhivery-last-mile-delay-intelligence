
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

sys.path.insert(0, str(PROJECT_ROOT))

from src.shipment_prep import clean_shipments, build_features

# ------------------------------------------------------------
# 1. LOAD NEW SHIPMENTS AND HUB MASTER
# ------------------------------------------------------------

new_shipments = pd.read_csv(
    PROJECT_ROOT / "data" / "new_shipments.csv"
)

hubs = pd.read_csv(
    PROJECT_ROOT / "data" / "hubs.csv"
)

segments = pd.read_csv(
    OUTPUT_DIR / "hub_segments.csv"
)

assert len(new_shipments) == 48
assert new_shipments["shipment_id"].is_unique
assert segments["hub_id"].is_unique

# ------------------------------------------------------------
# 2. LOAD PACKAGED MODEL AND CONFIGURATION
# ------------------------------------------------------------

model = joblib.load(MODEL_DIR / "delay_model.pkl")

threshold = float(
    pd.read_csv(MODEL_DIR / "delay_threshold.csv")
    .loc[0, "threshold"]
)

feature_order = (
    pd.read_csv(MODEL_DIR / "feature_order.csv")
    ["feature"]
    .tolist()
)

median_table = pd.read_csv(
    MODEL_DIR / "impute_values.csv"
)

medians = dict(
    zip(median_table["column"], median_table["median"])
)

assert len(feature_order) == 17
assert set(medians) == {
    "distance_km",
    "parcel_weight_kg",
    "declared_value_inr"
}

# ------------------------------------------------------------
# 3. CLEAN AND BUILD FEATURES
# ------------------------------------------------------------

cleaned, _ = clean_shipments(
    new_shipments,
    medians=medians
)

features_df = build_features(cleaned, hubs)

assert len(features_df) == len(new_shipments)
assert features_df["shipment_id"].tolist() == (
    new_shipments["shipment_id"].tolist()
)

X_new = features_df[feature_order].copy()

assert X_new.notna().sum().sum() == X_new.size

# ------------------------------------------------------------
# 4. PREDICT DELAY PROBABILITIES
# ------------------------------------------------------------

raw_probabilities = model.predict_proba(X_new)[:, 1]

# Use full-precision probabilities for the threshold decision
flagged = (raw_probabilities >= threshold).astype(int)

# Round probabilities only for the reporting output
reported_probabilities = np.round(raw_probabilities, 4)

# ------------------------------------------------------------
# 5. ATTACH HUB SEGMENTS
# ------------------------------------------------------------

actions = features_df[["shipment_id", "hub_id"]].copy()

actions["delay_probability"] = reported_probabilities
actions["flagged"] = flagged

actions = actions.merge(
    segments[["hub_id", "segment"]],
    on="hub_id",
    how="left",
    sort=False,
    validate="many_to_one"
)

assert actions["segment"].notna().all()

# ------------------------------------------------------------
# 6. ASSIGN OPERATIONAL ACTIONS
# ------------------------------------------------------------

actions["action"] = np.select(
    [
        (actions["flagged"] == 1) &
        (actions["segment"] == "Escalate"),

        (actions["flagged"] == 1)
    ],
    [
        "Escalate to hub manager",
        "Proactive customer call"
    ],
    default="No action"
)

# Required output column order
actions = actions[
    [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action"
    ]
]

# ------------------------------------------------------------
# 7. ACCEPTANCE CHECKS
# ------------------------------------------------------------

assert len(actions) == 48
assert actions["shipment_id"].is_unique
assert actions.isna().sum().sum() == 0

assert actions["shipment_id"].tolist() == (
    new_shipments["shipment_id"].tolist()
)

assert np.array_equal(
    actions["flagged"].to_numpy(),
    (raw_probabilities >= threshold).astype(int)
)

assert (
    actions.loc[actions["flagged"] == 0, "action"]
    == "No action"
).all()

assert (
    actions.loc[
        (actions["flagged"] == 1) &
        (actions["segment"] == "Escalate"),
        "action"
    ] == "Escalate to hub manager"
).all()

assert (
    actions.loc[
        (actions["flagged"] == 1) &
        (actions["segment"] != "Escalate"),
        "action"
    ] == "Proactive customer call"
).all()

# ------------------------------------------------------------
# 8. SAVE OUTPUT
# ------------------------------------------------------------

output_path = OUTPUT_DIR / "new_shipment_actions.csv"

actions.to_csv(
    output_path,
    index=False,
    float_format="%.4f"
)

print("=" * 65)
print("PART D — NEW SHIPMENT SCORING")
print("=" * 65)

print("\nFIRST 10 SCORED SHIPMENTS")
print(actions.head(10).to_string(index=False))

print("\nACTION SUMMARY")
print(actions["action"].value_counts().to_string())

print("\nSEGMENT SUMMARY")
print(
    actions.groupby("segment")
    .agg(
        shipments=("shipment_id", "count"),
        flagged=("flagged", "sum")
    )
    .to_string()
)

print("\nACCEPTANCE CHECKS")
print("48 shipments scored: PASS")
print("Unique shipment IDs: PASS")
print("Original shipment order preserved: PASS")
print("No missing output values: PASS")
print("Threshold decisions verified: PASS")
print("Operational actions verified: PASS")

print("\nOutput:", output_path)
print("PASS: New shipment scoring completed.")
