"""Descriptive evidence and fixed staged models, separate from hyperparameter search.

Descriptions use all listings. The staged prediction comparison uses only the
development partition and the same folds as tuning. No stage sees test prices.
"""
from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from threadpoolctl import threadpool_limits

import config as cfg
from tune_models import build_estimator, predict_prices, metrics


def save_table(table, output, name):
    """Keep chart evidence available as plain CSV files."""
    table.to_csv(output / "tables" / f"{name}.csv", index=False)


def segment_summary(data, columns):
    """Report price AND vehicle mix; raw segment differences are not causal."""
    result = data.groupby(columns, observed=True, dropna=False).agg(
        n=(cfg.TARGET, "size"), median_price=(cfg.TARGET, "median"),
        mean_price=(cfg.TARGET, "mean"), asking_value=(cfg.TARGET, "sum"),
        q25_price=(cfg.TARGET, lambda s: s.quantile(.25)),
        q75_price=(cfg.TARGET, lambda s: s.quantile(.75)),
        median_age=("Vehicle_Age", "median"), median_mileage=("Mileage", "median"),
        minimum_count=("At_Price_Floor", "sum")).reset_index()
    result["listing_share_pct"] = 100 * result.n / len(data)
    result["asking_value_share_pct"] = 100 * result.asking_value / data[cfg.TARGET].sum()
    result["minimum_rate_pct"] = 100 * result.minimum_count / result.n
    result["small_sample"] = result.n < cfg.SMALL_GROUP_N
    # Wilson interval: uncertainty around a proportion, assuming independent rows.
    z = NormalDist().inv_cdf((1 + cfg.CONFIDENCE_LEVEL) / 2)
    p, n = result.minimum_count / result.n, result.n
    centre = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    result["minimum_rate_low_pct"] = 100*(centre-half)
    result["minimum_rate_high_pct"] = 100*(centre+half)
    return result


def descriptive_analysis(df, output):
    """Summarise the supplied data; nothing here fits a pricing model."""
    profiles = []
    for field in cfg.SEGMENTS:
        table = segment_summary(df, [field]).rename(columns={field: "segment"})
        table.insert(0, "dimension", field)
        profiles.append(table)
    segments = pd.concat(profiles, ignore_index=True)
    save_table(segments, output, "segment_profiles")
    age = segment_summary(df, ["Vehicle_Age"])
    save_table(age, output, "age_profile")
    display = df.copy()
    display["Age_Band"] = pd.cut(display.Vehicle_Age, cfg.AGE_EDGES, labels=cfg.AGE_LABELS)
    display["Mileage_Band"] = pd.cut(display.Mileage, cfg.MILEAGE_EDGES, labels=cfg.MILEAGE_LABELS, right=False)
    age_bands = segment_summary(display, ["Age_Band"])
    grid = segment_summary(display, ["Age_Band", "Mileage_Band"])
    save_table(age_bands, output, "age_band_asking_value")
    save_table(grid, output, "minimum_by_age_and_mileage")
    missing = []
    for field in cfg.REQUIRED_COLUMNS:
        unknown = df[field].isna() | df[field].eq("Unknown")
        for label, mask in [("Known", ~unknown), ("Missing/Unknown", unknown)]:
            subset = df.loc[mask]
            missing.append({"field": field, "status": label, "n": len(subset),
                            "median_price": subset[cfg.TARGET].median(),
                            "median_age": subset.Vehicle_Age.median(),
                            "median_mileage": subset.Mileage.median()})
    save_table(pd.DataFrame(missing), output, "missingness_comparison")
    correlations = []
    for population, sample in [("All", df), ("Above recorded minimum", df[df[cfg.TARGET] > cfg.RECORDED_PRICE_MINIMUM])]:
        for first, second in [(c, cfg.TARGET) for c in cfg.CORRELATION_COLUMNS] + [("Vehicle_Age", "Mileage")]:
            pair = sample[[first, second]].dropna()
            if len(pair) >= cfg.CORRELATION_MINIMUM_N and pair.nunique().min() > 1:
                correlations.append({"population": population, "first": first, "second": second,
                                     "n": len(pair), "pearson": pearsonr(pair[first], pair[second]).statistic,
                                     "spearman": spearmanr(pair[first], pair[second]).statistic})
    save_table(pd.DataFrame(correlations), output, "correlations")
    efficiency = []
    for fuel, part in df.groupby("Fuel_Type"):
        pair = part[["Fuel_Efficiency", cfg.TARGET]].dropna()
        valid = len(pair) >= cfg.CORRELATION_MINIMUM_N and pair.nunique().min() > 1
        efficiency.append({"fuel_type": fuel, "n_known": len(pair),
                           "median_source_units": pair.Fuel_Efficiency.median(),
                           "spearman_with_price": spearmanr(pair.Fuel_Efficiency, pair[cfg.TARGET]).statistic if valid else np.nan,
                           "units_confirmed": False})
    save_table(pd.DataFrame(efficiency), output, "efficiency_within_fuel")
    return {"segments": segments, "age": age, "age_bands": age_bands, "minimum_grid": grid}


def staged_comparison(development, folds, output):
    """Use fixed reference settings to explain what each feature block adds."""
    stages = [
        ("Median baseline", "median", cfg.FULL_FEATURES),
        ("Age only (linear)", "linear", ["Vehicle_Age"]),
        ("Mileage only (linear)", "linear", ["Mileage"]),
        ("Age + mileage (linear)", "linear", cfg.AGE_MILEAGE),
        ("Age + mileage (curved)", "ridge", cfg.AGE_MILEAGE),
        ("Add model", "ridge", cfg.AGE_MILEAGE + ["Model"]),
        ("Add condition", "ridge", cfg.AGE_MILEAGE + ["Model"] + cfg.CONDITION_FEATURES),
        ("Add other attributes", "ridge", cfg.FULL_FEATURES),
    ]
    # Also remove blocks from the full specification. These are predictive
    # contributions under this specification, not causal feature effects.
    removals = [("Without " + name, "ridge", [c for c in cfg.FULL_FEATURES if c not in block])
                for name, block in [("model", ["Model"]), ("condition", cfg.CONDITION_FEATURES),
                                    ("other attributes", cfg.OTHER_FEATURES)]]
    y = development[cfg.TARGET].to_numpy()
    rows = []
    with threadpool_limits(limits=cfg.NUMERICAL_THREADS):
        for label, family, features in stages + removals:
            pred = np.full(len(development), np.nan)
            parameters = cfg.STAGED_RIDGE if family == "ridge" else {}
            for train, validation in folds:
                model = build_estimator(family, parameters, features)
                model.fit(development.iloc[train][features], y[train])
                pred[validation] = predict_prices(model, development.iloc[validation][features], parameters)
            rows.append({"stage": label, "population": "Development only", **metrics(y, pred)})
    result = pd.DataFrame(rows)
    save_table(result, output, "fixed_staged_models")
    return result.iloc[:len(stages)].copy()


def charts(df, desc, stages, output):
    """Make readable charts backed by the exported numerical tables."""
    blue, orange, grey = cfg.CHART_COLOURS
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    def finish(fig, name, caption):
        fig.text(.03, .015, caption, fontsize=8, color=grey)
        fig.tight_layout(rect=[0, .06, 1, 1])
        fig.savefig(output / "charts" / f"{name}.png", dpi=cfg.CHART_DPI)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(df[cfg.TARGET], bins=cfg.HISTOGRAM_BINS, color=blue)
    axes[0].axvline(df[cfg.TARGET].median(), color=orange, label="Median")
    axes[0].axvline(df[cfg.TARGET].mean(), color=grey, linestyle="--", label="Mean")
    axes[0].set(xlabel="Listed price ($)", ylabel="Listings", title="Listed-price distribution")
    axes[0].legend()
    age = desc["age"]
    axes[1].plot(age.Vehicle_Age, age.median_price, color=blue, label="Median")
    axes[1].fill_between(age.Vehicle_Age, age.q25_price, age.q75_price, color=blue, alpha=.2, label="Middle 50%")
    axes[1].set(xlabel="Vehicle age (assumed reference year)", ylabel="Listed price ($)", title="Price and age")
    axes[1].legend()
    finish(fig, "01_prices_and_age", "All listings. Shading is the middle half of observed prices, not a confidence interval.")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(stages.stage, stages.mae, color=blue)
    ax.invert_yaxis()
    ax.set(xlabel="Development out-of-fold MAE ($)", title="What information helps the fixed model?")
    finish(fig, "02_staged_models", "Same development folds. Fixed reference settings; no test rows; unbounded Ridge is diagnostic.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, field in zip(axes, [cfg.TARGET, "Vehicle_Age"]):
        density = ax.hexbin(df.Mileage, df[field], gridsize=cfg.HEXBIN_CELLS, bins="log", mincnt=1, cmap="Blues")
        fig.colorbar(density, ax=ax, label="Listing density (log scale)")
        ax.set(xlabel="Mileage", ylabel=field)
    finish(fig, "03_mileage_relationships", "Dense regions describe the sample. These patterns are not causal effects.")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    age_bands = desc["age_bands"]
    x = np.arange(len(age_bands))
    ax.bar(x-.2, age_bands.listing_share_pct, width=.4, label="Share of listings", color=blue)
    ax.bar(x+.2, age_bands.asking_value_share_pct, width=.4, label="Share of asking value", color=orange)
    ax.set_xticks(x, age_bands.Age_Band.astype(str))
    ax.set(xlabel="Vehicle age band", ylabel="Share (%)", title="Where recorded asking value sits")
    ax.legend()
    finish(fig, "04_asking_value", "Sum of asking prices is not capital invested, realised revenue, or profit.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, field in zip(axes, ["Accident_Status", "Service_History"]):
        part = desc["segments"][desc["segments"].dimension.eq(field)]
        ax.bar(part.segment, part.median_price, color=blue)
        ax.set(title=field.replace("_", " "), ylabel="Median listed price ($)")
    finish(fig, "05_condition", "Unadjusted differences include vehicle mix; Unknown is an information state.")


def run_analysis(df, development, folds, output: Path):
    """Called by the runner after model selection is frozen and tested."""
    for folder in [output, output / "tables", output / "charts"]:
        folder.mkdir(parents=True, exist_ok=True)
    desc = descriptive_analysis(df, output)
    stages = staged_comparison(development, folds, output)
    charts(df, desc, stages, output)
