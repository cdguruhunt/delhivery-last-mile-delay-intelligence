
from pathlib import Path
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# 1. PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

from src.shipment_prep import FEATURES, NOISE_COLUMNS

df = pd.read_csv(PROJECT_ROOT / "data/modelling_table.csv")

X = df[FEATURES].copy()
y = df["is_delayed"].astype(int).copy()

print("=" * 65)
print("PART C — DELAY CLASSIFICATION")
print("=" * 65)

# ============================================================
# 2. STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

assert (len(X_train), len(X_val), len(X_test)) == (840, 180, 180)
assert (int(y_train.sum()), int(y_val.sum()), int(y_test.sum())) == (
    195, 42, 41
)

print("\nPASS: Stratified split")

# ============================================================
# 3. CLASSIFICATION METRICS HELPER
# ============================================================

def classification_metrics(model_name, actual, predictions):

    tn, fp, fn, tp = confusion_matrix(
        actual, predictions, labels=[0, 1]
    ).ravel()

    return {
        "model": model_name,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy_score(actual, predictions),
        "precision": precision_score(
            actual, predictions, zero_division=0
        ),
        "recall": recall_score(
            actual, predictions, zero_division=0
        ),
        "f1": f1_score(
            actual, predictions, zero_division=0
        )
    }

# ============================================================
# 4. BASELINE AND CLASSIFIER COMPARISON
# ============================================================

results = []

baseline_predictions = np.zeros(len(y_val), dtype=int)

results.append(
    classification_metrics(
        "blind_all_on_time", y_val, baseline_predictions
    )
)

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

logistic_model.fit(X_train, y_train)

results.append(
    classification_metrics(
        "logistic_regression",
        y_val,
        logistic_model.predict(X_val)
    )
)

tree_depth_results = []

for depth in [2, 3, 4, 5, 6, 8, 10]:

    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    predictions = tree.predict(X_val)

    tree_depth_results.append({
        "depth": depth,
        "val_f1": f1_score(
            y_val, predictions, zero_division=0
        )
    })

best_tree = max(
    tree_depth_results,
    key=lambda row: (row["val_f1"], -row["depth"])
)

best_depth = best_tree["depth"]

tree_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=best_depth,
    random_state=42
)

tree_model.fit(X_train, y_train)

results.append(
    classification_metrics(
        f"decision_tree_depth_{best_depth}",
        y_val,
        tree_model.predict(X_val)
    )
)

default_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

default_rf.fit(X_train, y_train)

results.append(
    classification_metrics(
        "random_forest_default",
        y_val,
        default_rf.predict(X_val)
    )
)

models_df = pd.DataFrame(results)

models_df.to_csv(
    OUTPUT_DIR / "part_c_models.csv",
    index=False,
    float_format="%.4f"
)

print("\nCLASSIFIER COMPARISON")
print(models_df.to_string(index=False))

print("\nSelected decision-tree depth:", best_depth)

# ============================================================
# 5. FEATURE SCORES AND RANDOM FOREST IMPORTANCE
# ============================================================

rf_all = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_all.fit(X_train, y_train)

selector_all = SelectKBest(
    score_func=f_classif,
    k="all"
)

selector_all.fit(X_train, y_train)

f_scores = pd.Series(selector_all.scores_, index=FEATURES)

f_ranks = f_scores.rank(
    method="min",
    ascending=False
).astype(int)

rf_importances = pd.Series(
    rf_all.feature_importances_,
    index=FEATURES
)

feature_scores_df = pd.DataFrame({
    "feature": FEATURES,
    "f_score": [f_scores[f] for f in FEATURES],
    "f_rank": [f_ranks[f] for f in FEATURES],
    "rf_importance": [rf_importances[f] for f in FEATURES],
    "is_noise": [f in NOISE_COLUMNS for f in FEATURES]
})

feature_scores_df = feature_scores_df.sort_values(
    "f_rank"
).reset_index(drop=True)

feature_scores_df.to_csv(
    OUTPUT_DIR / "part_c_feature_scores.csv",
    index=False,
    float_format="%.4f"
)

# ============================================================
# 6. FEATURE SELECTION COMPARISON
# ============================================================

selection_results = []

for k in [5, 7, 10, 12, 17]:

    model = Pipeline([
        ("selector", SelectKBest(f_classif, k=k)),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    selection_results.append({
        "k": k,
        "val_f1": f1_score(
            y_val, predictions, zero_division=0
        )
    })

selectk_df = pd.DataFrame(selection_results)

selectk_df.to_csv(
    OUTPUT_DIR / "part_c_selectk.csv",
    index=False,
    float_format="%.4f"
)

# ============================================================
# 7. HYPERPARAMETER TUNING ON TOP 7 FEATURES
# ============================================================

selector_7 = SelectKBest(f_classif, k=7)

X_train_7 = selector_7.fit_transform(X_train, y_train)
X_val_7 = selector_7.transform(X_val)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5
)

grid_search.fit(X_train_7, y_train)

grid_val_f1 = f1_score(
    y_val,
    grid_search.best_estimator_.predict(X_val_7),
    zero_division=0
)

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=9,
    cv=5,
    random_state=42
)

random_search.fit(X_train_7, y_train)

random_val_f1 = f1_score(
    y_val,
    random_search.best_estimator_.predict(X_val_7),
    zero_division=0
)

default_7 = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

default_7.fit(X_train_7, y_train)

default_val_f1 = f1_score(
    y_val,
    default_7.predict(X_val_7),
    zero_division=0
)

tuning_df = pd.DataFrame([
    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 135,
        "best_params": str(grid_search.best_params_),
        "best_cv_accuracy": grid_search.best_score_,
        "val_f1": grid_val_f1
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 45,
        "best_params": str(random_search.best_params_),
        "best_cv_accuracy": random_search.best_score_,
        "val_f1": random_val_f1
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": np.nan,
        "val_f1": default_val_f1
    }
])

tuning_df.to_csv(
    OUTPUT_DIR / "part_c_tuning.csv",
    index=False,
    float_format="%.4f"
)

# ============================================================
# 8. SELECT FINAL MODEL USING VALIDATION F1
# Tie priority: default, grid, randomized
# ============================================================

candidates = [
    ("default_forest_same_7_features", default_val_f1, {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1
    }),
    ("grid_search", grid_val_f1, grid_search.best_params_),
    ("randomized_search", random_val_f1, random_search.best_params_)
]

chosen_name, chosen_f1, chosen_params = sorted(
    candidates,
    key=lambda item: (-item[1], [
        "default_forest_same_7_features",
        "grid_search",
        "randomized_search"
    ].index(item[0]))
)[0]

final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(
        random_state=42,
        **chosen_params
    ))
])

final_model.fit(X_train, y_train)

print("\nSelected final model:", chosen_name)
print("Validation F1:", round(chosen_f1, 4))

# ============================================================
# 9. VALIDATION ALERT THRESHOLD COMPARISON
# ============================================================

val_probabilities = final_model.predict_proba(X_val)[:, 1]

threshold_results = []

for threshold in [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9
]:

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    threshold_results.append({
        "threshold": threshold,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_val, predictions, zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    OUTPUT_DIR / "part_c_thresholds.csv",
    index=False,
    float_format="%.4f"
)

eligible = threshold_df[threshold_df["recall"] >= 0.80]

chosen_threshold = (
    eligible["threshold"].max() if len(eligible) else 0.1
)

# ============================================================
# 10. ONE FINAL TEST EVALUATION
# ============================================================

test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= chosen_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, test_predictions, labels=[0, 1]
).ravel()

final_test_df = pd.DataFrame([{
    "threshold": chosen_threshold,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp,
    "accuracy": accuracy_score(y_test, test_predictions),
    "precision": precision_score(
        y_test, test_predictions, zero_division=0
    ),
    "recall": recall_score(
        y_test, test_predictions, zero_division=0
    ),
    "f1": f1_score(
        y_test, test_predictions, zero_division=0
    )
}])

final_test_df.to_csv(
    OUTPUT_DIR / "part_c_final_test.csv",
    index=False,
    float_format="%.4f"
)

# ============================================================
# 11. FINAL CONFIGURATION
# ============================================================

final_config_df = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": chosen_params["n_estimators"],
    "max_depth": (
        chosen_params["max_depth"]
        if chosen_params["max_depth"] is not None
        else "unlimited"
    ),
    "min_samples_leaf": chosen_params["min_samples_leaf"],
    "threshold": chosen_threshold,
    "chosen_from": chosen_name
}])

final_config_df.to_csv(
    OUTPUT_DIR / "part_c_final_config.csv",
    index=False
)

# ============================================================
# 12. FINAL ACCEPTANCE CHECKS
# ============================================================

assert len(models_df) == 4
assert len(feature_scores_df) == 17
assert len(selectk_df) == 5
assert len(tuning_df) == 3
assert len(threshold_df) == 9
assert len(final_test_df) == 1
assert len(final_config_df) == 1

assert chosen_name == "default_forest_same_7_features"
assert chosen_threshold == 0.5

assert int(final_test_df.iloc[0]["TN"]) == 125
assert int(final_test_df.iloc[0]["FP"]) == 14
assert int(final_test_df.iloc[0]["FN"]) == 10
assert int(final_test_df.iloc[0]["TP"]) == 31

print("\n" + "=" * 65)
print("ALL PART C OUTPUTS REGENERATED SUCCESSFULLY")
print("=" * 65)

print("\nGenerated files:")
for filename in [
    "part_c_models.csv",
    "part_c_feature_scores.csv",
    "part_c_selectk.csv",
    "part_c_tuning.csv",
    "part_c_thresholds.csv",
    "part_c_final_test.csv",
    "part_c_final_config.csv"
]:
    print(" - outputs/" + filename)

print("\nPASS: Standalone Part C classification pipeline completed.")
