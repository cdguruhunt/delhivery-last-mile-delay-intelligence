from pathlib import Path
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.shipment_prep import FEATURES


def calculate_metrics(model_name, split, y_actual, y_predicted):
    mse = mean_squared_error(y_actual, y_predicted)
    return {
        "model": model_name,
        "split": split,
        "MAE": mean_absolute_error(y_actual, y_predicted),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_actual, y_predicted)
    }


def main():
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    # 1. Load modelling table and select the pinned features.
    df = pd.read_csv(PROJECT_ROOT / "data/modelling_table.csv")
    X = df[FEATURES].copy()
    y = df["delivery_hours"].copy()

    # 2. Exact training, validation and test split.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # 3. Scaling check: fit only on training rows.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_scaled_max_abs_mean = np.abs(
        X_train_scaled.mean(axis=0)
    ).max()

    test_scaled_max_abs_mean = np.abs(
        X_test_scaled.mean(axis=0)
    ).max()

    checks = pd.DataFrame({
        "metric": [
            "train_rows",
            "val_rows",
            "test_rows",
            "train_scaled_max_abs_mean",
            "test_scaled_max_abs_mean"
        ],
        "value": [
            len(X_train),
            len(X_val),
            len(X_test),
            train_scaled_max_abs_mean,
            test_scaled_max_abs_mean
        ]
    })

    checks.to_csv(
        outputs_dir / "part_b_checks.csv",
        index=False,
        float_format="%.10f"
    )

    assert (len(X_train), len(X_val), len(X_test)) == (840, 180, 180)
    assert train_scaled_max_abs_mean < 1e-9
    assert test_scaled_max_abs_mean > 0.01

    # 4. Mean baseline.
    baseline_prediction = np.full(len(y_val), y_train.mean())

    results = [
        calculate_metrics(
            "mean_baseline", "val", y_val, baseline_prediction
        )
    ]

    # 5. Decision-tree depth diagnostic.
    depths = [1, 2, 3, 4, 6, 8, 12, None]
    tree_results = []

    for depth in depths:
        tree = DecisionTreeRegressor(
            max_depth=depth,
            random_state=42
        )
        tree.fit(X_train, y_train)

        train_prediction = tree.predict(X_train)
        val_prediction = tree.predict(X_val)

        tree_results.append({
            "max_depth": "unlimited" if depth is None else depth,
            "train_rmse": np.sqrt(
                mean_squared_error(y_train, train_prediction)
            ),
            "val_rmse": np.sqrt(
                mean_squared_error(y_val, val_prediction)
            )
        })

    tree_depth_df = pd.DataFrame(tree_results)
    tree_depth_df.to_csv(
        outputs_dir / "part_b_tree_depth.csv",
        index=False,
        float_format="%.4f"
    )

    best_tree_row = min(
        tree_results,
        key=lambda row: (
            row["val_rmse"],
            float("inf") if row["max_depth"] == "unlimited"
            else row["max_depth"]
        )
    )

    best_depth = best_tree_row["max_depth"]
    assert best_depth != "unlimited"

    # 6. Four candidate models.
    models = {
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "decision_tree": DecisionTreeRegressor(
            max_depth=best_depth,
            random_state=42
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        results.append(
            calculate_metrics(model_name, "val", y_val, predictions)
        )

    metrics_df = pd.DataFrame(results)

    # 7. Gradient Boosting learning-rate experiment.
    learning_rate_results = []

    for lr in [0.01, 0.1, 1.0]:
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=lr,
            random_state=42
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        learning_rate_results.append({
            "learning_rate": lr,
            "val_rmse": np.sqrt(
                mean_squared_error(y_val, predictions)
            ),
            "val_r2": r2_score(y_val, predictions)
        })

    learning_rate_df = pd.DataFrame(learning_rate_results)
    learning_rate_df.to_csv(
        outputs_dir / "part_b_gb_learning_rate.csv",
        index=False,
        float_format="%.4f"
    )

    learning_r2 = learning_rate_df.set_index(
        "learning_rate"
    )["val_r2"]

    assert learning_r2.loc[0.1] > learning_r2.loc[1.0]
    assert learning_r2.loc[1.0] > learning_r2.loc[0.01]

    # 8. Cross-validation of the two best validation models.
    ranked = metrics_df[
        metrics_df["model"] != "mean_baseline"
    ].sort_values("RMSE")

    top_two_names = ranked.head(2)["model"].tolist()

    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)

    # One shared KFold object for both models.
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_models = {
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
    }

    cv_results = []

    for model_name in top_two_names:
        scores = cross_val_score(
            cv_models[model_name],
            X_train_val,
            y_train_val,
            cv=cv
        )

        cv_results.append({
            "model": model_name,
            "mean_r2": scores.mean(),
            "std_r2": scores.std()
        })

    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(
        outputs_dir / "part_b_cv.csv",
        index=False,
        float_format="%.4f"
    )

    # 9. Final model: lowest validation RMSE.
    # Reuse the model already fitted on training data.
    winner_name = ranked.iloc[0]["model"]
    final_model = models[winner_name]

    # The only test-set prediction/evaluation in this script.
    test_predictions = final_model.predict(X_test)

    test_result = calculate_metrics(
        winner_name, "test", y_test, test_predictions
    )

    metrics_df = pd.concat(
        [metrics_df, pd.DataFrame([test_result])],
        ignore_index=True
    )

    metrics_df.to_csv(
        outputs_dir / "part_b_metrics.csv",
        index=False,
        float_format="%.4f"
    )

    # 10. Acceptance checks.
    assert len(tree_depth_df) == 8
    assert len(metrics_df) == 6
    assert (metrics_df["split"] == "val").sum() == 5
    assert (metrics_df["split"] == "test").sum() == 1
    assert len(learning_rate_df) == 3
    assert len(cv_df) == 2

    assert np.all(
        np.abs(metrics_df["RMSE"] - np.sqrt(metrics_df["MSE"])) < 0.001
    )

    assert np.all(
        np.diff(tree_depth_df["train_rmse"].to_numpy()) <= 1e-8
    )

    assert (cv_df["mean_r2"] >= 0.90).all()
    assert (cv_df["std_r2"] <= 0.03).all()

    print("PART B REGRESSION PIPELINE COMPLETED")
    print("-----------------------------------")
    print("Split sizes:", len(X_train), len(X_val), len(X_test))
    print("Best tree depth:", best_depth)
    print("Best validation model:", winner_name)
    print("Validation RMSE:", round(float(ranked.iloc[0]["RMSE"]), 4))
    print("Final test RMSE:", round(float(test_result["RMSE"]), 4))
    print("Final test R2:", round(float(test_result["R2"]), 4))
    print("\\nAll Part B acceptance checks passed.")
    print("Outputs saved in:", outputs_dir)


if __name__ == "__main__":
    main()
