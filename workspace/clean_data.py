#!/usr/bin/env python3
"""Prepare the used-vehicle dataset without deleting rows or inventing values.

Requires Python 3.10+: python -m pip install -r requirements.txt
Run: python clean_data.py --input dataset.csv --output dataset_clean.csv

The input CSV is never overwritten. A JSON quality report is written beside
the cleaned CSV. Import clean_data() if you prefer to work in a notebook.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


# All cleaning assumptions and column rules live in the central configuration.
from config import (
    REFERENCE_YEAR as DEFAULT_REFERENCE_YEAR,
    RECORDED_PRICE_MINIMUM as DEFAULT_PRICE_FLOOR,
    EARLIEST_VEHICLE_YEAR, NUMERIC_COLUMNS, TEXT_COLUMNS, REQUIRED_COLUMNS,
    CORE_COLUMNS, UNKNOWN_COLUMNS, DERIVED_COLUMNS, ALLOWED_CATEGORIES,
)


def validate_data(df: pd.DataFrame, reference_year: int) -> None:
    """Reject contradictions instead of silently deleting or repairing rows."""
    errors = []

    for column in CORE_COLUMNS:
        count = int(df[column].isna().sum())
        if count:
            errors.append(f"{column}: {count} missing core values")

    for column in NUMERIC_COLUMNS:
        # Missing numeric information is allowed; infinity is not.
        known = df[column].dropna()
        if not np.isfinite(known.to_numpy(dtype=float)).all():
            errors.append(f"{column}: contains infinite values")

    for column in ["Year", "Owners"]:
        known = df[column].dropna()
        if (known % 1 != 0).any():
            errors.append(f"{column}: must contain whole numbers")

    if ((df["Year"] < EARLIEST_VEHICLE_YEAR) | (df["Year"] > reference_year)).any():
        errors.append(f"Year: must be between {EARLIEST_VEHICLE_YEAR} and {reference_year}")
    if (df["Owners"] < 1).any():
        errors.append("Owners: must be at least 1")
    if (df["Mileage"] < 0).any():
        errors.append("Mileage: cannot be negative")
    if (df["Engine_Size"] < 0).any():
        errors.append("Engine_Size: cannot be negative")
    for column in ["Horsepower", "Torque", "Fuel_Efficiency", "Selling_Price"]:
        if (df[column].dropna() <= 0).any():
            errors.append(f"{column}: known values must be positive")

    accident_values = set(df["Accident_History"].dropna().unique())
    if not accident_values.issubset({0, 1}):
        errors.append("Accident_History: only 0, 1, or missing is supported")

    for column, allowed in ALLOWED_CATEGORIES.items():
        unexpected = set(df[column].dropna().unique()) - allowed
        if unexpected:
            errors.append(f"{column}: unexpected categories {sorted(unexpected)}")

    zero_ice = df["Engine_Size"].eq(0) & df["Fuel_Type"].ne("Electric")
    if zero_ice.any():
        errors.append(f"Engine_Size: {int(zero_ice.sum())} non-EVs have zero size")
    model_make_counts = df.groupby("Model")["Make"].nunique()
    if (model_make_counts > 1).any():
        errors.append("Model: at least one model maps to multiple makes")

    if errors:
        raise ValueError("Data validation failed:\n- " + "\n- ".join(errors))


def clean_data(
    raw: pd.DataFrame,
    reference_year: int = DEFAULT_REFERENCE_YEAR,
    price_floor: float = DEFAULT_PRICE_FLOOR,
) -> tuple[pd.DataFrame, dict]:
    """Return a new cleaned DataFrame and an auditable quality summary.

    Rows and their order are preserved. Numeric missing values remain missing.
    The raw DataFrame is not changed, and extra input columns are retained.
    """
    if raw.empty:
        raise ValueError("The input dataset has no rows.")
    if not raw.columns.is_unique:
        raise ValueError("Duplicate column names are not supported.")
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Required columns are missing: {missing_columns}")
    collisions = sorted(set(DERIVED_COLUMNS) & set(raw.columns))
    if collisions:
        raise ValueError(f"Input already has derived columns: {collisions}")
    if not np.isfinite(price_floor) or price_floor <= 0:
        raise ValueError("price_floor must be finite and positive.")
    if not isinstance(reference_year, int) or reference_year < EARLIEST_VEHICLE_YEAR:
        raise ValueError(f"reference_year must be an integer of at least {EARLIEST_VEHICLE_YEAR}.")

    df = raw.copy(deep=True)

    # Strip accidental whitespace. An empty string means missing information.
    # Do not alter capitalization or merge different categories without evidence.
    for column in TEXT_COLUMNS:
        df[column] = df[column].astype("string").str.strip().replace("", pd.NA)

    # Convert numeric strings, but never use errors='coerce': that could hide
    # corrupted values by turning them into ordinary missing observations.
    for column in NUMERIC_COLUMNS:
        blank = df[column].map(lambda value: isinstance(value, str) and not value.strip())
        df[column] = df[column].mask(blank, np.nan)
        try:
            df[column] = pd.to_numeric(df[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{column}: contains a non-numeric value") from exc

    validate_data(df, reference_year)
    missing_before = {column: int(df[column].isna().sum()) for column in df}
    complete_rows = int(df[REQUIRED_COLUMNS].notna().all(axis=1).sum())
    duplicate_rows = int(df.duplicated().sum())

    # Missing category != a known negative. For example, unknown service history
    # must not be relabeled 'No Service'. Leave numeric missing values untouched.
    for column in UNKNOWN_COLUMNS:
        df[column] = df[column].fillna("Unknown")

    # Preserve the original 0/1/NaN accident column alongside a readable version.
    df["Accident_Status"] = (
        df["Accident_History"].map({0: "No", 1: "Yes"}).fillna("Unknown")
    )
    df["Vehicle_Age"] = (reference_year - df["Year"]).astype("int64")

    # Equality flags the observed pile-up. It does NOT prove censoring, imply
    # profitability, or justify changing/deleting the vehicle's recorded price.
    df["At_Price_Floor"] = df["Selling_Price"].eq(price_floor)

    # Units are undocumented. Flag EVs for separate efficiency analysis without
    # claiming their unit is MPGe or converting the raw efficiency values.
    df["Efficiency_Requires_Separate_Analysis"] = df["Fuel_Type"].eq("Electric")

    notes = [
        "All rows retained; numeric missing values not imputed.",
        "Reference year is an analytical assumption, not a confirmed listing date.",
        "Price-floor flag records exact equality; censoring remains a hypothesis.",
        "EV efficiency scale may differ; units need source confirmation.",
        "Extreme mileage retained; a statistical outlier is not proof of error.",
        "No arbitrary age/mileage bands or winsorization applied.",
    ]
    if duplicate_rows:
        notes.append("Duplicate rows retained: no vehicle ID to establish duplicates.")
    below_floor = int(df["Selling_Price"].lt(price_floor).sum())
    if below_floor:
        notes.append("Prices below the configured floor exist; review the floor assumption.")

    report = {
        "reference_year": reference_year,
        "price_floor": price_floor,
        "input_rows": len(raw),
        "output_rows": len(df),
        "input_columns": len(raw.columns),
        "output_columns": len(df.columns),
        "complete_rows_across_required_columns": complete_rows,
        "duplicate_rows_after_whitespace_normalization": duplicate_rows,
        "missing_before_treatment": missing_before,
        "missing_after_treatment": {col: int(df[col].isna().sum()) for col in df},
        "at_price_floor": int(df["At_Price_Floor"].sum()),
        "at_price_floor_percent": round(100 * df["At_Price_Floor"].mean(), 2),
        "below_configured_floor": below_floor,
        "electric_vehicles": int(df["Fuel_Type"].eq("Electric").sum()),
        "zero_engine_size_vehicles": int(df["Engine_Size"].eq(0).sum()),
        "maximum_mileage": float(df["Mileage"].max()),
        "notes": notes,
    }
    return df, report


def main() -> None:
    """Read, validate, clean, and save using command-line file paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset_clean.csv"))
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    parser.add_argument("--reference-year", type=int, default=DEFAULT_REFERENCE_YEAR)
    parser.add_argument("--price-floor", type=float, default=DEFAULT_PRICE_FLOOR)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    report_path = (args.report or output_path.with_suffix(".quality.json")).resolve()

    # Avoid accidental data loss, even if --overwrite was supplied.
    if len({input_path, output_path, report_path}) != 3:
        parser.error("Input, cleaned CSV, and report must have different paths.")
    if not args.overwrite:
        for path in [output_path, report_path]:
            if path.exists():
                parser.error(f"Output already exists: {path}. Use --overwrite to replace it.")

    try:
        raw = pd.read_csv(input_path, encoding="utf-8-sig")
        cleaned, report = clean_data(raw, args.reference_year, args.price_floor)
        report["input_file"] = str(input_path)
        report["output_file"] = str(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        # Blank CSV cells represent numeric missingness; no index column added.
        cleaned.to_csv(output_path, index=False, encoding="utf-8", na_rep="")
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    logging.info("Rows preserved: %s -> %s", report["input_rows"], report["output_rows"])
    logging.info("Price-floor observations: %s (%.2f%%)",
                 report["at_price_floor"], report["at_price_floor_percent"])
    logging.info("Cleaned CSV: %s", output_path)
    logging.info("Quality report: %s", report_path)


if __name__ == "__main__":
    main()
