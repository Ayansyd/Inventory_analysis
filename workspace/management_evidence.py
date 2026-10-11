#!/usr/bin/env python3
"""Rebuild management-slide evidence after run_analysis.py finishes.

All inventory descriptions use every listing. Queue counts come from the new
development-only audit, with its population labelled explicitly. This script
does not train or select a model. No old queue snapshot is copied forward.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from tune_models import write_json


def build_evidence(results: Path):
    """Use this run's clean data and review counts for traceable slide numbers."""
    source = results / "dataset_clean.csv"
    queue_file = results / "decision" / "queue_composition.csv"
    if not source.is_file() or not queue_file.is_file():
        raise ValueError("Run run_analysis.py first; this export needs its clean data and decision outputs.")
    # Do not silently apply changed assumptions to an old run's cleaned data.
    if (results / "config_used.py").read_bytes() != (cfg.ROOT / "config.py").read_bytes():
        raise ValueError("config.py differs from this run's snapshot. Restore it or run a new experiment first.")
    output = results / "management"
    if output.exists() and any(output.iterdir()):
        raise ValueError("Management output already exists. Use a fresh results directory.")
    df = pd.read_csv(source)
    if not np.allclose(df.Vehicle_Age, cfg.REFERENCE_YEAR-df.Year):
        raise ValueError("Cleaned ages disagree with the configured reference year.")
    output.mkdir(parents=True, exist_ok=True)
    df["Age_Band"] = pd.cut(df.Vehicle_Age, cfg.AGE_EDGES, labels=cfg.AGE_LABELS)
    age = df.groupby("Age_Band", observed=True).agg(n=(cfg.TARGET, "size"), asking_value=(cfg.TARGET, "sum")).reset_index()
    age["listing_share_pct"] = 100*age.n/len(df)
    age["asking_value_share_pct"] = 100*age.asking_value/df[cfg.TARGET].sum()
    age.to_csv(output / "age_concentration.csv", index=False)
    ratios = {}
    for field in ["Make", "Body_Type"]:
        rows = []
        for segment, part in df.groupby(field):
            young = part.loc[part.Age_Band.eq(cfg.AGE_LABELS[0]), cfg.TARGET]
            older = part.loc[part.Age_Band.eq(cfg.AGE_LABELS[1]), cfg.TARGET]
            valid = len(young) > 0 and len(older) > 0
            rows.append({"segment": segment, "young_age_band": cfg.AGE_LABELS[0],
                         "older_age_band": cfg.AGE_LABELS[1], "n_young": len(young), "n_older": len(older),
                         "median_young": float(young.median()) if len(young) else None,
                         "median_older": float(older.median()) if len(older) else None,
                         "older_to_younger_price_ratio_pct": 100*float(older.median()/young.median()) if valid else None,
                         "small_group": min(len(young), len(older)) < cfg.SMALL_GROUP_N})
        table = pd.DataFrame(rows).sort_values("older_to_younger_price_ratio_pct", ascending=False)
        table.to_csv(output / f"{field.lower()}_ratios.csv", index=False)
        ratios[field] = rows
    minimum = df[cfg.TARGET].eq(cfg.RECORDED_PRICE_MINIMUM)
    premium = df.Make.isin(cfg.PREMIUM_MAKES)
    summary = {"n": len(df), "total_asking_value": float(df[cfg.TARGET].sum()),
               "premium_n": int(premium.sum()), "premium_listing_share_pct": float(100*premium.mean()),
               "premium_asking_value_share_pct": float(100*df.loc[premium, cfg.TARGET].sum()/df[cfg.TARGET].sum()),
               "recorded_minimum_n": int(minimum.sum()), "recorded_minimum_share_pct": float(100*minimum.mean()),
               "reference_year": cfg.REFERENCE_YEAR, "queue_population": "Development partition only",
               "ratio_interpretation": "Different cars in broad age groups; not observed depreciation or a stocking recommendation.",
               "asking_value_interpretation": "Sum of listed prices, not invested capital, realised revenue, or profit."}
    write_json(output / "slide_data.json", {"summary": summary,
        "age": age.to_dict("records"), "ratios": ratios,
        "queues": pd.read_csv(queue_file).to_dict("records"),
        "clean_data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "queue_source_sha256": hashlib.sha256(queue_file.read_bytes()).hexdigest(),
        "export_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(f"Management evidence saved to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=cfg.OUTPUT_DIRECTORY)
    args = parser.parse_args()
    build_evidence(args.results.resolve())


if __name__ == "__main__":
    main()
