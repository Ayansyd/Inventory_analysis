#!/usr/bin/env python3
"""Run cleaning, one bounded tuning experiment, analysis, and review outputs.

Usage from this directory: python run_analysis.py
Choose another output folder: python run_analysis.py --output results_second_run
Edit analytical settings in config.py. Existing results are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import logging
import platform
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as cfg
from clean_data import clean_data
from tune_models import (candidates, make_split, search_candidates, select_candidate,
                         evaluate_frozen_choice, write_report, write_json)
from Analysis import run_analysis
from Decision_analysis import run_decisions

LOG = logging.getLogger(__name__)


def validate_configuration():
    """Catch inconsistent edits before any expensive model fitting starts."""
    if not 0 < cfg.TEST_FRACTION < 1 or cfg.CV_FOLDS < 2:
        raise ValueError("TEST_FRACTION must be between 0 and 1; CV_FOLDS must be at least 2.")
    if cfg.PRIMARY_METRIC != "MAE":
        raise ValueError("This implementation selects by MAE. Changing its name does not change the scoring code.")
    if cfg.SELECTION_STANDARD_ERRORS < 0 or not 0 < cfg.CONFIDENCE_LEVEL < 1:
        raise ValueError("Invalid selection or confidence setting.")
    if not 0 < cfg.ERROR_PERCENTILE < 1:
        raise ValueError("ERROR_PERCENTILE must lie between zero and one.")
    if cfg.REVIEW_PER_PRICE_BAND < 1 or cfg.BOOTSTRAP_REPLICATES < 2:
        raise ValueError("Review quota and bootstrap count must be positive.")
    if cfg.MATCH_MINIMUM_PER_GROUP < 1 or cfg.MATCH_MILEAGE_WIDTH <= 0 or any(w <= 0 for w in cfg.MATCH_AGE_WIDTHS):
        raise ValueError("Invalid comparable-group settings.")
    if len(cfg.FULL_FEATURES) != len(set(cfg.FULL_FEATURES)):
        raise ValueError("Predictor names must be unique.")
    if set(cfg.FULL_FEATURES) & {cfg.TARGET, "At_Price_Floor", "Source_Record_Number", "Mileage_At_Recorded_Minimum"}:
        raise ValueError("Outcome-derived flags and record identifiers must not be model inputs.")
    if cfg.TARGET in cfg.DUPLICATE_GROUP_COLUMNS:
        raise ValueError("Duplicate grouping must not use the target.")
    for edges, labels in [(cfg.AGE_EDGES, cfg.AGE_LABELS), (cfg.MILEAGE_EDGES, cfg.MILEAGE_LABELS),
                          (cfg.PRICE_EDGES, cfg.PRICE_LABELS)]:
        if len(edges) != len(labels)+1 or any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError("Band edges must increase, with one more edge than labels.")
    plan = candidates()
    for family, parameters in [("ridge", cfg.RIDGE_REFERENCE), ("forest", cfg.FOREST_REFERENCE)]:
        if not any(c["family"] == family and c["parameters"] == parameters for c in plan):
            raise ValueError(f"Keep the fixed {family} reference inside its search grid.")


def main():
    """Orchestrate steps, keeping the reserved test out of candidate selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=cfg.INPUT_FILE)
    parser.add_argument("--output", type=Path, default=cfg.OUTPUT_DIRECTORY)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    validate_configuration()
    if not source.is_file():
        parser.error(f"Input CSV not found: {source}")
    if source.is_relative_to(output):
        parser.error("Keep the input outside the output directory.")
    if output.exists() and any(output.iterdir()):
        parser.error("Output folder is not empty. Choose a new --output; results are never overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log", encoding="utf-8")])
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "threadpoolctl", "joblib"]
    versions = {name: importlib.metadata.version(name) for name in packages}
    manifest = {"status": "started", "started_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "packages": versions,
                "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(cfg.ROOT.glob("*.py"))}}
    write_json(output / "run_manifest.json", manifest)
    (output / "config_used.py").write_bytes((cfg.ROOT / "config.py").read_bytes())
    (output / "requirements_used.txt").write_text("# Exact versions for Python " + platform.python_version() + "\n" +
        "\n".join(f"{name}=={version}" for name, version in versions.items())+"\n", encoding="utf-8")
    try:
        LOG.info("1/5: Clean and validate the raw data")
        raw = pd.read_csv(source, encoding="utf-8-sig")
        df, quality = clean_data(raw)
        df["Source_Record_Number"] = np.arange(1, len(df)+1)
        df.to_csv(output / "dataset_clean.csv", index=False)
        write_json(output / "quality_report.json", quality)
        tuning_dir = output / "tuning"
        tuning_dir.mkdir()
        development, test, folds = make_split(df, tuning_dir)
        LOG.info("Reserved %d rows; %d development rows. Test is not passed into search.", len(test), len(development))
        LOG.info("2/5: Evaluate the fixed search grid")
        scores, plan, cache = search_candidates(development, folds, tuning_dir)
        selection = select_candidate(scores, plan, tuning_dir)
        LOG.info("3/5: Choice frozen: %s. Evaluate the reserved test once.", selection["selected"])
        test_scores, development_predictions = evaluate_frozen_choice(development, test, selection, plan, cache, tuning_dir)
        write_report(scores, selection, test_scores, tuning_dir)
        LOG.info("4/5: Export descriptions and fixed staged models")
        run_analysis(df, development, folds, output / "exploratory")
        LOG.info("5/5: Build the development-only review audit and decision memo")
        run_decisions(df, development, development_predictions, test_scores, selection, output / "decision")
        manifest.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat(),
                        selected_candidate=selection["selected"],
                        statement="Fixed search complete; no post-test retuning; test rows excluded from model fitting and review-queue construction.")
        write_json(output / "run_manifest.json", manifest)
        LOG.info("Complete. Read %s", tuning_dir / "TUNING_REPORT.md")
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        write_json(output / "run_manifest.json", manifest)
        LOG.exception("Run stopped; completed search candidates remain in search_results.csv.")
        raise


if __name__ == "__main__":
    main()
