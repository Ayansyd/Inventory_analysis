#!/usr/bin/env python3
"""Estimate listed prices with a saved experimental model, without retraining.

Only load a joblib file produced by your own run; pickle-based files can execute
code when loaded. Input is a raw, complete-schema listing CSV including its
current listed price. That price is NEVER passed to the estimator.
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

import config as cfg
from clean_data import clean_data
from tune_models import predict_prices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() in {args.input.resolve(), args.model.resolve()}:
        parser.error("Choose a new output file distinct from the input and saved model.")
    bundle = joblib.load(args.model)
    raw = pd.read_csv(args.input, encoding="utf-8-sig")
    data, _ = clean_data(raw, reference_year=bundle["reference_year"],
                         price_floor=bundle["parameters"].get("prediction_minimum") or cfg.RECORDED_PRICE_MINIMUM)
    result = raw.copy()
    result["Estimated_Listed_Price"] = predict_prices(bundle["estimator"], data[bundle["features"]], bundle["parameters"])
    result["Listed_Minus_Estimated"] = data[cfg.TARGET] - result.Estimated_Listed_Price
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result):,} experimental estimates to {args.output}")


if __name__ == "__main__":
    main()
