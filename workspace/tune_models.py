"""One bounded search, a frozen choice, and one reserved-test evaluation.

The test partition is never passed to search_candidates() or select_candidate().
All fitted preprocessing is rebuilt inside each training fold. Settings and
search ranges come from config.py. This module does no work on import.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, KFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

import config as cfg

LOG = logging.getLogger(__name__)


def write_json(path: Path, data: dict | list) -> None:
    """Write readable metadata, rejecting non-standard JSON numbers."""
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def make_split(df: pd.DataFrame, output: Path):
    """Reserve about 20% of description groups, then build development folds.

    Identical recorded vehicle descriptions stay together. This cannot identify
    the same real vehicle when its description has changed; a VIN is still needed.
    Group splitting means the exact row fraction can differ slightly from 20%.
    """
    groups = pd.util.hash_pandas_object(df[cfg.DUPLICATE_GROUP_COLUMNS], index=False)
    split = GroupShuffleSplit(n_splits=1, test_size=cfg.TEST_FRACTION,
                              random_state=cfg.RANDOM_SEED)
    dev_idx, test_idx = next(split.split(df, groups=groups))
    development = df.iloc[dev_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    dev_groups = groups.iloc[dev_idx].to_numpy()
    unique_groups = np.unique(dev_groups)
    if len(unique_groups) < cfg.CV_FOLDS:
        raise ValueError("Not enough independent description groups for the configured folds.")
    splitter = KFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_SEED)
    folds = []
    fold_labels = np.zeros(len(development), dtype=int)
    for number, (train_g, val_g) in enumerate(splitter.split(unique_groups), start=1):
        train = np.flatnonzero(np.isin(dev_groups, unique_groups[train_g]))
        val = np.flatnonzero(np.isin(dev_groups, unique_groups[val_g]))
        folds.append((train, val))
        fold_labels[val] = number
    assert not set(groups.iloc[dev_idx]) & set(groups.iloc[test_idx])
    assignments = pd.concat([
        pd.DataFrame({"Source_Record_Number": development.Source_Record_Number,
                      "partition": "development", "validation_fold": fold_labels}),
        pd.DataFrame({"Source_Record_Number": test.Source_Record_Number,
                      "partition": "reserved_test", "validation_fold": 0}),
    ], ignore_index=True).sort_values("Source_Record_Number")
    assignments.to_csv(output / "split_assignments.csv", index=False)
    write_json(output / "split_summary.json", {
        "development_rows": len(development), "test_rows": len(test),
        "description_groups": int(groups.nunique()),
        "rows_beyond_unique_descriptions": int(len(df) - groups.nunique()),
        "test_row_fraction": len(test) / len(df), "random_seed": cfg.RANDOM_SEED,
        "limitation": "Internally reserved for this search; this dataset was previously explored. Not external or temporal validation.",
    })
    return development, test, folds


def build_estimator(family: str, parameters: dict, features=None):
    """Build an unfitted pipeline. All learning happens later inside fit()."""
    features = list(cfg.FULL_FEATURES if features is None else features)
    if family == "median":
        return DummyRegressor(strategy="median")
    if family == "linear":
        return LinearRegression()
    numeric = [c for c in features if c in cfg.NUMERIC_FEATURES]
    categories = [c for c in features if c not in numeric]
    transforms = []
    if family == "ridge":
        curved = [c for c in cfg.AGE_MILEAGE if c in features]
        other_numeric = [c for c in numeric if c not in curved]
        if curved:
            transforms.append(("curves", Pipeline([
                ("spline", SplineTransformer(n_knots=parameters["n_knots"],
                    degree=parameters["degree"], **cfg.SPLINE_FIXED)),
                ("scale", StandardScaler()),
            ]), curved))
        if other_numeric:
            transforms.append(("numeric", StandardScaler(), other_numeric))
        estimator = Ridge(alpha=parameters["alpha"], **cfg.RIDGE_FIXED)
    elif family == "forest":
        if numeric:
            transforms.append(("numeric", "passthrough", numeric))
        estimator = RandomForestRegressor(**cfg.FOREST_FIXED, **parameters)
    else:
        raise ValueError(f"Unknown model family: {family}")
    if categories:
        transforms.append(("categories", OneHotEncoder(**cfg.ENCODER_SETTINGS), categories))
    return Pipeline([("prepare", ColumnTransformer(transforms)), ("estimate", estimator)])


def predict_prices(estimator, x: pd.DataFrame, parameters: dict) -> np.ndarray:
    """Apply a configured lower bound consistently in CV, test, and inference."""
    prediction = np.asarray(estimator.predict(x), dtype=float)
    minimum = parameters.get("prediction_minimum")
    return np.maximum(prediction, minimum) if minimum is not None else prediction


def metrics(y, prediction) -> dict:
    """Use dollar errors; R2 is supporting information, not percent accuracy."""
    y, prediction = np.asarray(y), np.asarray(prediction)
    error = y - prediction
    return {"n": len(y), "mae": float(np.abs(error).mean()),
            "median_absolute_error": float(np.median(np.abs(error))),
            "p90_absolute_error": float(np.quantile(np.abs(error), cfg.ERROR_PERCENTILE)),
            "r2": float(r2_score(y, prediction)) if len(y) > 1 else None,
            "mean_actual_minus_predicted": float(error.mean()),
            "negative_predictions": int((prediction < 0).sum())}


def candidates() -> list[dict]:
    """Enumerate the entire fixed search before seeing any validation scores."""
    result = [{"family": "median", "parameters": {}, "selectable": True}]
    for family, grid in [("ridge", cfg.RIDGE_GRID), ("forest", cfg.FOREST_GRID)]:
        for parameters in ParameterGrid(grid):
            result.append({"family": family, "parameters": parameters,
                           "selectable": family != "ridge" or parameters["prediction_minimum"] is not None})
    for index, item in enumerate(result, start=1):
        item["candidate_id"] = f"C{index:03d}"
    return result


def search_candidates(development: pd.DataFrame, folds, output: Path):
    """Evaluate every registered candidate once; never expand or chase scores.

    The CSV checkpoint is rewritten after each completed candidate. Every row
    contains its exact settings and every fold's validation and training MAE.
    Cached predictions are development diagnostics, not independent test scores.
    """
    plan = candidates()
    write_json(output / "search_plan.json", {
        "candidate_count": len(plan), "folds": cfg.CV_FOLDS,
        "total_cv_fits": len(plan) * cfg.CV_FOLDS,
        "metric": cfg.PRIMARY_METRIC, "candidates": plan,
        "stopping_rule": "Exhaust this fixed grid once; freeze choice; evaluate reserved test once; no expansion.",
    })
    rows, cache = [], {}
    x = development[cfg.FULL_FEATURES]
    y = development[cfg.TARGET].to_numpy(dtype=float)
    with threadpool_limits(limits=cfg.NUMERICAL_THREADS):
        for index, candidate in enumerate(plan, start=1):
            start = time.perf_counter()
            params, family = candidate["parameters"], candidate["family"]
            LOG.info("Candidate %d/%d | %s | %s", index, len(plan), family, params)
            oof = np.full(len(y), np.nan)
            fold_scores, train_scores = [], []
            for train, val in folds:
                estimator = build_estimator(family, params)
                estimator.fit(x.iloc[train], y[train])
                oof[val] = predict_prices(estimator, x.iloc[val], params)
                fold_scores.append(mean_absolute_error(y[val], oof[val]))
                train_scores.append(mean_absolute_error(y[train], predict_prices(estimator, x.iloc[train], params)))
            if not np.isfinite(oof).all():
                raise ValueError(f"Incomplete/nonfinite predictions for {candidate['candidate_id']}")
            row = {"candidate_id": candidate["candidate_id"], "family": family,
                   "selectable": candidate["selectable"], "parameters_json": json.dumps(params, sort_keys=True),
                   **params, "cv_mean_mae": float(np.mean(fold_scores)),
                   "cv_std_mae": float(np.std(fold_scores, ddof=1)),
                   "cv_se_mae": float(np.std(fold_scores, ddof=1) / math.sqrt(cfg.CV_FOLDS)),
                   "train_mean_mae": float(np.mean(train_scores)),
                   "oof_mae": metrics(y, oof)["mae"], "oof_r2": metrics(y, oof)["r2"],
                   "negative_oof_predictions": int((oof < 0).sum()),
                   "elapsed_seconds": time.perf_counter() - start}
            row["validation_minus_training_mae"] = row["cv_mean_mae"] - row["train_mean_mae"]
            for number, (validation, training) in enumerate(zip(fold_scores, train_scores), start=1):
                row[f"fold_{number}_mae"] = float(validation)
                row[f"fold_{number}_train_mae"] = float(training)
            rows.append(row)
            cache[candidate["candidate_id"]] = oof
            pd.DataFrame(rows).to_csv(output / "search_results.csv", index=False)
            LOG.info("  CV MAE $%.2f | train MAE $%.2f", row["cv_mean_mae"], row["train_mean_mae"])
    return pd.DataFrame(rows), plan, cache


def simplicity(candidate: dict) -> tuple:
    """Predeclared tie order: median, Ridge, forest; then lower complexity.

    Ridge: fewer knots, lower degree, stronger regularisation.
    Forest: shallower trees, larger leaves, fewer trees, fewer features per split.
    This is a decision convention, not a mathematical ranking of all models.
    """
    p = candidate["parameters"]
    if candidate["family"] == "median":
        return (0,)
    if candidate["family"] == "ridge":
        return (1, p["n_knots"], p["degree"], -p["alpha"])
    return (2, p["max_depth"] if p["max_depth"] is not None else float("inf"),
            -p["min_samples_leaf"], p["n_estimators"], p["max_features"])


def select_candidate(scores: pd.DataFrame, plan: list[dict], output: Path):
    """Select using development CV only; write the frozen choice before testing."""
    eligible = scores.loc[scores.selectable].sort_values(["cv_mean_mae", "candidate_id"])
    best = eligible.iloc[0]
    threshold = float(best.cv_mean_mae + cfg.SELECTION_STANDARD_ERRORS * best.cv_se_mae)
    admissible_ids = set(eligible.loc[eligible.cv_mean_mae <= threshold, "candidate_id"])
    selected = min((c for c in plan if c["candidate_id"] in admissible_ids), key=simplicity)
    selected_score = scores.set_index("candidate_id").loc[selected["candidate_id"]]
    selection = {"selected": selected,
                 "lowest_cv_mae_candidate": next(c for c in plan if c["candidate_id"] == best.candidate_id),
                 "lowest_cv_mae": float(best.cv_mean_mae),
                 "selected_cv_mae": float(selected_score.cv_mean_mae),
                 "selection_threshold_mae": threshold,
                 "eligible_within_threshold": sorted(admissible_ids),
                 "rule": "Simplest eligible candidate within one configured CV standard error of the lowest MAE.",
                 "uncertainty_note": "Fold standard errors are heuristic; overlapping training sets violate independence.",
                 "claim": "Best tested CV candidate and a simplicity-based recommendation; no claim of global optimality.",
                 "test_used_for_selection": False}
    scores = scores.copy()
    scores["within_selection_threshold"] = scores.selectable & (scores.cv_mean_mae <= threshold)
    scores["selected"] = scores.candidate_id.eq(selected["candidate_id"])
    scores.to_csv(output / "search_results.csv", index=False)
    write_json(output / "selected_parameters.json", selection)
    return selection


def evaluate_frozen_choice(development, test, selection, plan, cache, output):
    """Fit on development only, score reserved rows, and save the chosen model.

    Evaluate predeclared reference models and the CV winner too. Their test
    scores do NOT change the recommended candidate. No model is refit on test.
    """
    comparisons = {"selected": selection["selected"],
                   "lowest_cv_mae": selection["lowest_cv_mae_candidate"],
                   "median_reference": next(c for c in plan if c["family"] == "median")}
    for family, reference in [("ridge", cfg.RIDGE_REFERENCE), ("forest", cfg.FOREST_REFERENCE)]:
        item = next((c for c in plan if c["family"] == family and c["parameters"] == reference), None)
        if item is None:
            raise ValueError(f"The {family} reference must remain in its search grid.")
        comparisons[family + "_reference"] = item
    fitted, predictions, rows, bands = {}, {}, [], []
    for role, candidate in comparisons.items():
        identifier = candidate["candidate_id"]
        if identifier not in fitted:
            estimator = build_estimator(candidate["family"], candidate["parameters"])
            with threadpool_limits(limits=cfg.NUMERICAL_THREADS):
                estimator.fit(development[cfg.FULL_FEATURES], development[cfg.TARGET])
                predictions[identifier] = predict_prices(estimator, test[cfg.FULL_FEATURES], candidate["parameters"])
            fitted[identifier] = estimator
        prediction = predictions[identifier]
        for population, mask in [("All reserved test", np.ones(len(test), dtype=bool)),
                                  ("Above minimum: same test model", test[cfg.TARGET].gt(cfg.RECORDED_PRICE_MINIMUM).to_numpy())]:
            if mask.any():
                rows.append({"role": role, "candidate_id": identifier, "population": population,
                             **metrics(test.loc[mask, cfg.TARGET], prediction[mask])})
        price_band = pd.cut(test[cfg.TARGET], cfg.PRICE_EDGES, labels=cfg.PRICE_LABELS)
        for band in cfg.PRICE_LABELS:
            mask = price_band.eq(band).to_numpy()
            if mask.any():
                bands.append({"role": role, "candidate_id": identifier, "price_band": band,
                              **metrics(test.loc[mask, cfg.TARGET], prediction[mask])})
    pd.DataFrame(rows).to_csv(output / "test_metrics.csv", index=False)
    pd.DataFrame(bands).to_csv(output / "test_errors_by_price_band.csv", index=False)
    chosen = selection["selected"]
    identifier = chosen["candidate_id"]
    oof = development[["Source_Record_Number", cfg.TARGET]].copy()
    oof["predicted_price"] = cache[identifier]
    oof["prediction_origin"] = "development CV; parameters selected using these folds"
    oof.to_csv(output / "selected_development_predictions.csv", index=False)
    holdout = test[["Source_Record_Number", cfg.TARGET]].copy()
    holdout["predicted_price"] = predictions[identifier]
    holdout["prediction_origin"] = "internally reserved test; selection frozen first"
    holdout.to_csv(output / "selected_test_predictions.csv", index=False)
    joblib.dump({"estimator": fitted[identifier], "parameters": chosen["parameters"],
                 "features": cfg.FULL_FEATURES, "candidate": chosen,
                 "reference_year": cfg.REFERENCE_YEAR,
                 "training_population": "development only; reserved test excluded"},
                output / "selected_model.joblib", compress=cfg.SAVE_MODEL_COMPRESSION)
    return pd.DataFrame(rows), oof


def write_report(scores, selection, test_scores, output):
    """Explain the entire experiment using its actual results, never placeholders."""
    best, chosen = selection["lowest_cv_mae_candidate"], selection["selected"]
    lookup = scores.set_index("candidate_id")
    lines = ["# Tuning report", "", "## What was actually tried", "",
             f"{len(scores)} candidates, each on the same {cfg.CV_FOLDS} development folds.",
             f"Ridge grid: `{json.dumps(cfg.RIDGE_GRID)}`",
             f"Forest grid: `{json.dumps(cfg.FOREST_GRID)}`",
             "Median prediction is the baseline. Unconstrained Ridge candidates are diagnostic only.",
             "`search_plan.json` enumerates every combination; `search_results.csv` contains every setting, fold MAE, training MAE, timing and selection flag.", "",
             "## Decision made before reading the test scores", "",
             f"Lowest eligible CV MAE: **{best['candidate_id']} {best['family']}**, ${selection['lowest_cv_mae']:,.2f}.",
             f"Its parameters: `{json.dumps(best['parameters'])}`", "",
             f"Recommended: **{chosen['candidate_id']} {chosen['family']}**, CV MAE ${selection['selected_cv_mae']:,.2f}.",
             f"Its parameters: `{json.dumps(chosen['parameters'])}`", "",
             f"The predeclared simplicity threshold was ${selection['selection_threshold_mae']:,.2f}.",
             "Within that threshold, prefer median, then Ridge, then forest; prefer fewer knots/lower degree/stronger Ridge regularisation or shallower trees/larger leaves/fewer trees. This is an approximate one-standard-error heuristic, not a significance test.", "",
             "## Reserved-test results (same frozen models)", "",
             "| Role | Population | MAE | R2 | 90th-percentile error |",
             "|---|---|---:|---:|---:|"]
    for row in test_scores.itertuples():
        r2 = f"{row.r2:.3f}" if pd.notna(row.r2) else "n/a"
        lines.append(f"| {row.role} | {row.population} | ${row.mae:,.2f} | {r2} | ${row.p90_absolute_error:,.2f} |")
    lines += ["", "Above-minimum rows are evaluated as a subset of the same test predictions, not with a separately refitted model.", "",
              "## Compared with the fixed starting settings", ""]
    for family, reference in [("ridge", cfg.RIDGE_REFERENCE), ("forest", cfg.FOREST_REFERENCE)]:
        reference_row = scores.loc[(scores.family == family) & scores.parameters_json.eq(json.dumps(reference, sort_keys=True))].iloc[0]
        family_best = scores.loc[(scores.family == family) & scores.selectable].sort_values("cv_mean_mae").iloc[0]
        delta = reference_row.cv_mean_mae - family_best.cv_mean_mae
        lines.append(f"- {family}: reference {reference_row.candidate_id} CV MAE ${reference_row.cv_mean_mae:,.2f}; best tested {family_best.candidate_id} ${family_best.cv_mean_mae:,.2f}; reduction ${delta:,.2f} ({100*delta/reference_row.cv_mean_mae:.2f}%).")
    chosen_row = lookup.loc[chosen["candidate_id"]]
    lines += ["", "## Why stop here?", "",
              "The fixed search is complete. No ranges, seeds, features, or model families are changed after reviewing scores. The recommendation remains frozen even if another comparator scores better on test.",
              f"Selected candidate training MAE ${chosen_row.train_mean_mae:,.2f}; validation MAE ${chosen_row.cv_mean_mae:,.2f}. This gap is a diagnostic, not a proof that all overfitting is absent.",
              "This is the best tested CV result within a declared budget, not a mathematically proven global optimum. More search is not automatically more trustworthy.", "",
              "## What can change in the presentation", "",
              "Replace the old 'settings were not tuned' statement with the fixed-search protocol and actual selected parameters. Use the test table here for this new experiment; do not substitute development MAE into a test-result claim.",
              "The previous full-dataset CV scores used different splits, training sizes and possibly solver details. They are historical context, not a paired estimate of tuning improvement. Use the fixed reference rows in this run for the comparison.",
              "Improved prediction does not establish profitable stock choices or prove that review flags are errors. No cost, realised-price or turnover data was added. Business claims about margin and stocking still require those data and a measured pilot.", "",
              "## Boundaries", "",
              "This dataset was previously explored. The test subset is reserved only from this new search, not from all prior human knowledge. Fresh listings with dates and vehicle IDs are needed for independent or temporal validation.",
              "Exact description groups are kept together; this cannot identify every repeat listing without VIN. Fold standard errors are approximate because folds share training observations.",
              "A $500 constraint remains an explicit recorded-price assumption. Human review uses development predictions selected on these folds, so it is retrospective and has no independently measured success rate.",
              "The saved model is trained on development rows only. It is an experimental listed-price estimator, not a production valuation model.", ""]
    (output / "TUNING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
