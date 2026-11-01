
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# 1. Load source data
# ---------------------------------------------------------

df = pd.read_csv(DATA_DIR / "modelling_table.csv")
hubs = pd.read_csv(DATA_DIR / "hubs.csv")

# ---------------------------------------------------------
# 2. Build hub profile
# ---------------------------------------------------------

hub_profile = (
    df.groupby("hub_id", as_index=False)
      .agg(
          shipments=("shipment_id", "count"),
          mean_delivery_hours=("delivery_hours", "mean"),
          delay_rate=("is_delayed", "mean"),
          mean_distance_km=("distance_km", "mean"),
          mean_n_stops=("n_stops", "mean")
      )
      .sort_values("hub_id")
      .reset_index(drop=True)
)

hub_profile.to_csv(OUTPUT_DIR / "hub_profile.csv", index=False)

assert len(hub_profile) == 24
assert hub_profile["hub_id"].is_unique
assert hub_profile["shipments"].eq(50).all()
assert hub_profile.isna().sum().sum() == 0
assert hub_profile["hub_id"].tolist() == sorted(hub_profile["hub_id"].tolist())

# ---------------------------------------------------------
# 3. Scale the four hub-level clustering features
# ---------------------------------------------------------

CLUSTER_FEATURES = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops"
]

X = hub_profile[CLUSTER_FEATURES].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------
# 4. Calculate separation evidence for K = 2, 3, 4, 5
# ---------------------------------------------------------

separation_rows = []

for k in [2, 3, 4, 5]:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    within_distances = []
    across_distances = []

    for i, j in itertools.combinations(range(len(X_scaled)), 2):

        distance = float(np.linalg.norm(X_scaled[i] - X_scaled[j]))

        if labels[i] == labels[j]:
            within_distances.append(distance)
        else:
            across_distances.append(distance)

    mean_within = np.mean(within_distances)
    mean_across = np.mean(across_distances)

    cluster_sizes = sorted(
        pd.Series(labels).value_counts().tolist(),
        reverse=True
    )

    separation_rows.append({
        "K": k,
        "across_over_within": mean_across / mean_within,
        "cluster_sizes": "-".join(map(str, cluster_sizes))
    })

separation = pd.DataFrame(separation_rows)
separation.to_csv(OUTPUT_DIR / "part_d_separation.csv", index=False)

assert len(separation) == 4
assert separation["K"].tolist() == [2, 3, 4, 5]
assert separation.loc[separation["K"] == 2, "cluster_sizes"].iloc[0] == "15-9"
assert separation.loc[separation["K"] == 3, "cluster_sizes"].iloc[0] == "9-8-7"
assert np.isclose(
    separation.loc[separation["K"] == 2, "across_over_within"].iloc[0],
    2.894,
    atol=0.01
)
assert np.isclose(
    separation.loc[separation["K"] == 3, "across_over_within"].iloc[0],
    5.393,
    atol=0.01
)

# ---------------------------------------------------------
# 5. Final K=3 hub segmentation
# ---------------------------------------------------------

final_model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

cluster_labels = final_model.fit_predict(X_scaled)

clustered = hub_profile.copy()
clustered["cluster_id"] = cluster_labels

# Rank clusters by their mean hub-level delay rate.
cluster_delay_rates = (
    clustered.groupby("cluster_id")["delay_rate"]
             .mean()
             .sort_values()
)

segment_names = {
    cluster_delay_rates.index[0]: "Standard",
    cluster_delay_rates.index[1]: "Watch",
    cluster_delay_rates.index[2]: "Escalate"
}

clustered["segment"] = clustered["cluster_id"].map(segment_names)

# ---------------------------------------------------------
# 6. Create hub_segments.csv
# ---------------------------------------------------------

hub_segments = (
    clustered[["hub_id", "segment"]]
    .merge(
        hubs[["hub_id", "hub_city", "hub_tier"]],
        on="hub_id",
        how="left",
        validate="one_to_one"
    )
    .sort_values("hub_id")
    .reset_index(drop=True)
)

hub_segments.to_csv(OUTPUT_DIR / "hub_segments.csv", index=False)

assert len(hub_segments) == 24
assert hub_segments["hub_id"].is_unique
assert hub_segments["segment"].notna().all()
assert hub_segments["hub_city"].notna().all()
assert hub_segments["hub_tier"].notna().all()

# ---------------------------------------------------------
# 7. Create segment_profile.csv
# ---------------------------------------------------------

segment_profile = (
    clustered.groupby("segment", as_index=False)
    .agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean")
    )
)

segment_order = ["Standard", "Watch", "Escalate"]

segment_profile["segment"] = pd.Categorical(
    segment_profile["segment"],
    categories=segment_order,
    ordered=True
)

segment_profile = (
    segment_profile.sort_values("segment")
                   .reset_index(drop=True)
)

segment_profile["segment"] = segment_profile["segment"].astype(str)

segment_profile.to_csv(
    OUTPUT_DIR / "segment_profile.csv",
    index=False
)

assert segment_profile["segment"].tolist() == segment_order
assert segment_profile["hubs"].tolist() == [7, 8, 9]

# ---------------------------------------------------------
# 8. Compare segment against expected hub-tier mapping
# ---------------------------------------------------------

tier_to_segment = {
    1: "Standard",
    2: "Watch",
    3: "Escalate"
}

hub_segments["expected_segment"] = (
    pd.to_numeric(hub_segments["hub_tier"], errors="coerce")
      .map(tier_to_segment)
)

mismatch = hub_segments[
    hub_segments["segment"] != hub_segments["expected_segment"]
][["hub_id", "segment", "hub_city", "hub_tier"]].copy()

mismatch.to_csv(
    OUTPUT_DIR / "tier_vs_segment_mismatch.csv",
    index=False
)

assert len(mismatch) == 4
assert mismatch["hub_id"].tolist() == [
    "HUB07", "HUB13", "HUB15", "HUB22"
]

# ---------------------------------------------------------
# 9. Display results and acceptance checks
# ---------------------------------------------------------

print("\nHUB PROFILE — REFERENCE SPOT CHECKS")
print(
    hub_profile[
        hub_profile["hub_id"].isin(["HUB01", "HUB13", "HUB17"])
    ].round(4).to_string(index=False)
)

print("\nK-MEANS SEPARATION EVIDENCE")
print(separation.round(4).to_string(index=False))

print("\nSEGMENT PROFILE")
print(segment_profile.round(4).to_string(index=False))

print("\nHUB SEGMENTS")
print(hub_segments.to_string(index=False))

print("\nTIER VS SEGMENT MISMATCH")
print(mismatch.to_string(index=False))

print("\nACCEPTANCE CHECKS")
print("Hub profile rows:", len(hub_profile))
print("Separation rows:", len(separation))
print("Segment counts:", segment_profile["hubs"].tolist())
print("Mismatch count:", len(mismatch))
print("Output files:")
print("- outputs/hub_profile.csv")
print("- outputs/part_d_separation.csv")
print("- outputs/hub_segments.csv")
print("- outputs/segment_profile.csv")
print("- outputs/tier_vs_segment_mismatch.csv")

print("\nPASS: Complete Part D segmentation pipeline executed successfully.")
