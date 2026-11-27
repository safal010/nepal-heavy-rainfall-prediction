
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/chirps/expanded")
OUTPUT_DIR = DATA_DIR / "analysis"

PRODUCTS = ["rnl", "sat"]

THRESHOLD_MM = 50.0

RAINFALL_FIELDS = {
    "center": "chirps_center_mm",
    "3x3_mean": "chirps_3x3_mean_mm",
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_data(product):

    filename = DATA_DIR / f"expanded_results_{product}.csv"

    if not filename.exists():
        raise FileNotFoundError(filename)

    df = pd.read_csv(filename)

    required_columns = [
        "date",
        "location",
        "nasa_rainfall_mm",
        "chirps_center_mm",
        "chirps_3x3_mean_mm",
        "heavy_nasa",
        "heavy_chirps_center",
        "heavy_chirps_3x3",
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{product.upper()} missing columns: {missing}"
        )

    df["date"] = pd.to_datetime(df["date"])

    if df[required_columns].isnull().any().any():
        raise ValueError(
            f"{product.upper()} contains missing values."
        )

    if df.duplicated(["date", "location"]).any():
        raise ValueError(
            f"{product.upper()} contains duplicate records."
        )

    # Check heavy-rainfall labels against the 50 mm threshold
    if not (
        df["heavy_nasa"].astype(int)
        == (df["nasa_rainfall_mm"] >= THRESHOLD_MM).astype(int)
    ).all():
        raise ValueError("NASA heavy-rainfall labels are inconsistent.")

    if not (
        df["heavy_chirps_center"].astype(int)
        == (df["chirps_center_mm"] >= THRESHOLD_MM).astype(int)
    ).all():
        raise ValueError("CHIRPS center labels are inconsistent.")

    if not (
        df["heavy_chirps_3x3"].astype(int)
        == (df["chirps_3x3_mean_mm"] >= THRESHOLD_MM).astype(int)
    ).all():
        raise ValueError("CHIRPS 3x3 labels are inconsistent.")

    return df


# ============================================================
# CALCULATE REGRESSION AND CLASSIFICATION METRICS
# ============================================================

def calculate_metrics(df, rainfall_column, heavy_column):

    actual = df["nasa_rainfall_mm"].to_numpy(dtype=float)
    predicted = df[rainfall_column].to_numpy(dtype=float)

    errors = predicted - actual
    absolute_errors = np.abs(errors)

    # Regression metrics
    mae = np.mean(absolute_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    bias = np.mean(errors)
    median_absolute_error = np.median(absolute_errors)

    if np.std(actual) > 0 and np.std(predicted) > 0:
        correlation = np.corrcoef(actual, predicted)[0, 1]
    else:
        correlation = np.nan

    # Classification metrics
    actual_heavy = (
        df["heavy_nasa"].to_numpy(dtype=int) == 1
    )

    predicted_heavy = (
        df[heavy_column].to_numpy(dtype=int) == 1
    )

    tp = int(np.sum(actual_heavy & predicted_heavy))
    tn = int(np.sum(~actual_heavy & ~predicted_heavy))
    fp = int(np.sum(~actual_heavy & predicted_heavy))
    fn = int(np.sum(actual_heavy & ~predicted_heavy))

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else np.nan

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    disagreement = (
        (fp + fn) / total * 100
        if total else np.nan
    )

    return {
        "records": total,

        "nasa_heavy_events": int(np.sum(actual_heavy)),

        "MAE_mm": mae,
        "RMSE_mm": rmse,
        "Bias_mm": bias,
        "Median_absolute_error_mm": median_absolute_error,
        "Pearson_correlation": correlation,

        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,

        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1_score": f1,
        "Disagreement_percent": disagreement,
    }


# ============================================================
# ANALYZE ONE DATA GROUP
# ============================================================

def analyze_group(df, product, group_name, group_value):

    rows = []

    for measurement, rainfall_column in RAINFALL_FIELDS.items():

        if measurement == "center":
            heavy_column = "heavy_chirps_center"
        else:
            heavy_column = "heavy_chirps_3x3"

        metrics = calculate_metrics(
            df,
            rainfall_column,
            heavy_column,
        )

        rows.append({

            "product": product.upper(),

            "measurement": measurement,

            "group": group_name,

            "group_value": group_value,

            **metrics,

        })

    return rows


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("\nEXPANDED CHIRPS VALIDATION ANALYSIS")

    print("=" * 65)

    all_data = {}
    overall_rows = []
    location_rows = []
    yearly_rows = []

    # Load both products
    for product in PRODUCTS:

        print(f"\nLoading {product.upper()} data...")

        df = load_data(product)

        print("Records:", len(df))
        print("Locations:", df["location"].nunique())
        print("Unique dates:", df["date"].nunique())

        all_data[product] = df

        # Overall metrics
        overall_rows.extend(
            analyze_group(
                df,
                product,
                "overall",
                "All records",
            )
        )

        # Location-wise metrics
        for location, group in df.groupby("location"):

            location_rows.extend(
                analyze_group(
                    group,
                    product,
                    "location",
                    location,
                )
            )

        # Year-wise metrics
        df["year"] = df["date"].dt.year

        for year, group in df.groupby("year"):

            yearly_rows.extend(
                analyze_group(
                    group,
                    product,
                    "year",
                    year,
                )
            )

    # Check paired records between RNL and SAT
    rnl_keys = set(
        zip(all_data["rnl"]["date"], all_data["rnl"]["location"])
    )

    sat_keys = set(
        zip(all_data["sat"]["date"], all_data["sat"]["location"])
    )

    if rnl_keys != sat_keys:
        raise ValueError(
            "RNL and SAT records do not match."
        )

    # Check that NASA observations match between products
    comparison = all_data["rnl"][
        ["date", "location", "nasa_rainfall_mm"]
    ].merge(
        all_data["sat"][
            ["date", "location", "nasa_rainfall_mm"]
        ],
        on=["date", "location"],
        suffixes=("_rnl", "_sat"),
        validate="one_to_one",
    )

    if not np.allclose(
        comparison["nasa_rainfall_mm_rnl"],
        comparison["nasa_rainfall_mm_sat"],
        rtol=0,
        atol=1e-9,
    ):
        raise ValueError(
            "NASA rainfall values differ between RNL and SAT."
        )

    # Convert results to DataFrames
    overall_df = pd.DataFrame(overall_rows)
    location_df = pd.DataFrame(location_rows)
    yearly_df = pd.DataFrame(yearly_rows)

    # Save results separately
    overall_df.to_csv(
        OUTPUT_DIR / "overall_metrics.csv",
        index=False,
    )

    location_df.to_csv(
        OUTPUT_DIR / "location_metrics.csv",
        index=False,
    )

    yearly_df.to_csv(
        OUTPUT_DIR / "yearly_metrics.csv",
        index=False,
    )

    # Print overall results
    print("\n" + "=" * 65)
    print("OVERALL VALIDATION RESULTS")
    print("=" * 65)

    display_columns = [
        "product",
        "measurement",
        "MAE_mm",
        "RMSE_mm",
        "Bias_mm",
        "Median_absolute_error_mm",
        "Pearson_correlation",
        "Accuracy",
        "Precision",
        "Recall",
        "F1_score",
        "Disagreement_percent",
    ]

    print(
        overall_df[display_columns].round(4).to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("ANALYSIS COMPLETED")
    print("=" * 65)

    print("\nSaved files:")

    print(OUTPUT_DIR / "overall_metrics.csv")
    print(OUTPUT_DIR / "location_metrics.csv")
    print(OUTPUT_DIR / "yearly_metrics.csv")


if __name__ == "__main__":
    main()
