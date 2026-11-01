
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

from src.shipment_prep import FEATURES, clean_shipments

print("=" * 65)
print("PART D — CLASSIFICATION MODEL PACKAGING")
print("=" * 65)

# ------------------------------------------------------------
# 1. LOAD FINAL PART C CONFIGURATION
# ------------------------------------------------------------

config = pd.read_csv(OUTPUT_DIR / "part_c_final_config.csv")

assert len(config) == 1

config_row = config.iloc[0]

selector_k = int(config_row["selector_k"])
n_estimators = int(config_row["n_estimators"])
max_depth_value = config_row["max_depth"]
min_samples_leaf = int(config_row["min_samples_leaf"])
threshold = float(config_row["threshold"])

max_depth = (
    None if str(max_depth_value).lower() == "unlimited"
    else int(max_depth_value)
)

assert selector_k == 7
assert n_estimators == 100
assert max_depth is None
assert min_samples_leaf == 1
assert threshold == 0.5

print("\nFinal configuration:")
print(config.to_string(index=False))

# ------------------------------------------------------------
# 2. REBUILD HISTORICAL IMPUTATION MEDIANS
# ------------------------------------------------------------

raw = pd.read_csv(PROJECT_ROOT / "data" / "raw_shipments.csv")

cleaned, medians = clean_shipments(raw)

median_df = pd.DataFrame([
    {"column": column, "median": value}
    for column, value in medians.items()
])

median_df.to_csv(
    MODEL_DIR / "impute_values.csv",
    index=False
)

print("\nHistorical imputation medians:")
print(median_df.to_string(index=False))

# ------------------------------------------------------------
# 3. REBUILD THE ORIGINAL PART C DATA SPLIT
# ------------------------------------------------------------

df = pd.read_csv(PROJECT_ROOT / "data" / "modelling_table.csv")

X = df[FEATURES].copy()
y = df["is_delayed"].astype(int).copy()

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

assert (len(X_train), len(X_val), len(X_test)) == (840, 180, 180)
assert (int(y_train.sum()), int(y_val.sum()), int(y_test.sum())) == (
    195, 42, 41
)

print("\nOriginal stratified split rebuilt: PASS")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))

# ------------------------------------------------------------
# 4. BUILD AND FIT THE FINAL PART C PIPELINE
# ------------------------------------------------------------

final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=selector_k)),
    ("model", RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    ))
])

final_model.fit(X_train, y_train)

# ------------------------------------------------------------
# 5. SAVE THE FOUR MODEL ARTEFACTS
# ------------------------------------------------------------

joblib.dump(
    final_model,
    MODEL_DIR / "delay_model.pkl"
)

pd.DataFrame({
    "threshold": [threshold]
}).to_csv(
    MODEL_DIR / "delay_threshold.csv",
    index=False
)

pd.DataFrame({
    "feature": FEATURES
}).to_csv(
    MODEL_DIR / "feature_order.csv",
    index=False
)

# ------------------------------------------------------------
# 6. RELOAD MODEL AND VERIFY PREDICTION EQUIVALENCE
# ------------------------------------------------------------

reloaded_model = joblib.load(
    MODEL_DIR / "delay_model.pkl"
)

original_probabilities = final_model.predict_proba(X_test)[:, 1]
reloaded_probabilities = reloaded_model.predict_proba(X_test)[:, 1]

max_probability_diff = float(
    np.max(
        np.abs(original_probabilities - reloaded_probabilities)
    )
)

reload_check = pd.DataFrame([
    {
        "metric": "test_rows_scored",
        "value": len(X_test)
    },
    {
        "metric": "reload_max_abs_probability_diff",
        "value": max_probability_diff
    },
    {
        "metric": "threshold",
        "value": threshold
    }
])

reload_check.to_csv(
    OUTPUT_DIR / "part_d_reload_check.csv",
    index=False
)

# ------------------------------------------------------------
# 7. ACCEPTANCE CHECKS
# ------------------------------------------------------------

required_files = [
    MODEL_DIR / "delay_model.pkl",
    MODEL_DIR / "delay_threshold.csv",
    MODEL_DIR / "feature_order.csv",
    MODEL_DIR / "impute_values.csv",
    OUTPUT_DIR / "part_d_reload_check.csv"
]

assert all(path.exists() for path in required_files)

saved_features = pd.read_csv(
    MODEL_DIR / "feature_order.csv"
)["feature"].tolist()

assert saved_features == FEATURES

saved_medians = pd.read_csv(
    MODEL_DIR / "impute_values.csv"
)

assert saved_medians["column"].tolist() == [
    "distance_km",
    "parcel_weight_kg",
    "declared_value_inr"
]

assert len(saved_medians) == 3
assert len(reload_check) == 3
assert len(X_test) == 180
assert max_probability_diff <= 1e-9

print("\nMODEL RELOAD CHECK")
print(reload_check.to_string(index=False))

print("\nGENERATED MODEL FILES")
for path in required_files:
    print("-", path.relative_to(PROJECT_ROOT))

print("\nACCEPTANCE CHECKS")
print("Original 840/180/180 split: PASS")
print("Exact 17-feature order: PASS")
print("Three historical medians saved: PASS")
print("Model saved and successfully reloaded: PASS")
print("180 test rows scored: PASS")
print("Reload probability difference <= 1e-9: PASS")

print("\nPASS: Model packaging completed successfully.")
