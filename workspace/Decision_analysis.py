"""Turn evidence into a human-review pilot, without claiming verified profit.

Audit candidates use development predictions only. The reserved test is kept
out of audit construction. Retrospective review ranks are not independently
validated probabilities of an incorrect listing.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
from tune_models import write_json


def build_review_outputs(development, predictions, output):
    """Compare ranking biases and allocate at most ten reviews per price band."""
    data = development.merge(predictions[["Source_Record_Number", "predicted_price"]],
                             on="Source_Record_Number", validate="one_to_one")
    if len(data) != len(development):
        raise ValueError("Every development record needs one held-out prediction.")
    data["residual"] = data[cfg.TARGET] - data.predicted_price
    data["absolute_error"] = data.residual.abs()
    data["percentage_gap"] = 100 * data.absolute_error / data[cfg.TARGET]
    data["price_band"] = pd.cut(data[cfg.TARGET], cfg.PRICE_EDGES, labels=cfg.PRICE_LABELS)
    if data.price_band.isna().any():
        raise ValueError("Some prices are outside the configured audit bands.")
    reference = data.groupby("price_band", observed=True).agg(
        n=("residual", "size"), median_residual=("residual", "median"),
        q25=("residual", lambda s: s.quantile(.25)),
        q75=("residual", lambda s: s.quantile(.75))).reset_index()
    overall_iqr = data.residual.quantile(.75) - data.residual.quantile(.25)
    if overall_iqr <= 0:
        raise ValueError("Residuals have no spread for a scaled audit ranking.")
    reference["scale"] = np.where((reference.n >= cfg.SMALL_GROUP_N) & (reference.q75 > reference.q25),
                                  reference.q75-reference.q25, overall_iqr)
    reference.to_csv(output / "review_band_reference.csv", index=False)
    data = data.merge(reference[["price_band", "median_residual", "scale"]], on="price_band", validate="many_to_one")
    data["band_centred_gap"] = (data.residual-data.median_residual).abs()/data.scale
    quota_total = cfg.REVIEW_PER_PRICE_BAND * len(cfg.PRICE_LABELS)
    queues = {"Dollar gap": data.sort_values(["absolute_error", "Source_Record_Number"], ascending=[False, True]).head(quota_total),
              "Percentage gap": data.sort_values(["percentage_gap", "Source_Record_Number"], ascending=[False, True]).head(quota_total)}
    selected, controls, quotas = [], [], []
    for band in cfg.PRICE_LABELS:
        cell = data[data.price_band.eq(band)]
        # Preserve disjoint groups if a future dataset has too few rows in a band.
        count = min(cfg.REVIEW_PER_PRICE_BAND, len(cell)//2)
        quotas.append({"price_band": band, "available": len(cell),
                       "requested_per_group": cfg.REVIEW_PER_PRICE_BAND, "actual_per_group": count})
        if not count:
            continue
        ranked = cell.sort_values(["band_centred_gap", "Source_Record_Number"], ascending=[False, True]).head(count)
        random = cell.loc[~cell.Source_Record_Number.isin(ranked.Source_Record_Number)].sample(n=count, random_state=cfg.RANDOM_SEED)
        selected.append(ranked)
        controls.append(random)
    if not selected:
        raise ValueError("Insufficient records to construct an audit and a control group.")
    queues["Balanced audit"] = pd.concat(selected, ignore_index=True)
    queues["Random controls"] = pd.concat(controls, ignore_index=True)
    assert not set(queues["Balanced audit"].Source_Record_Number) & set(queues["Random controls"].Source_Record_Number)
    pd.DataFrame(quotas).to_csv(output / "review_quotas.csv", index=False)
    summaries = []
    for name, queue in queues.items():
        queue = queue.copy()
        queue["review_outcome"] = ""
        queue["reviewer_notes"] = ""
        queue["review_minutes"] = ""
        queue.to_csv(output / (name.lower().replace(" ", "_") + ".csv"), index=False)
        summaries.append({"queue": name, "n": len(queue),
                          "premium_makes": int(queue.Make.isin(cfg.PREMIUM_MAKES).sum()),
                          "recorded_minimum": int(queue[cfg.TARGET].eq(cfg.RECORDED_PRICE_MINIMUM).sum()),
                          "listed_above_estimate": int(queue.residual.gt(0).sum())})
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "queue_composition.csv", index=False)
    # Give a reviewer shuffled records without revealing selection labels or gaps.
    blind = pd.concat([queues["Balanced audit"].assign(audit_group="selected"),
                       queues["Random controls"].assign(audit_group="random")], ignore_index=True)
    blind = blind.sample(frac=1, random_state=cfg.RANDOM_SEED).reset_index(drop=True)
    blind["Review_ID"] = np.arange(1, len(blind)+1)
    blind[["Review_ID", "Source_Record_Number", "audit_group"]].to_csv(output / "review_assignment_key.csv", index=False)
    sheet = blind[["Review_ID", "Source_Record_Number"]+cfg.REQUIRED_COLUMNS].copy()
    for field in ["verified_actionable", "issue_type", "review_minutes", "notes"]:
        sheet[field] = ""
    sheet.to_csv(output / "blinded_review_sheet.csv", index=False)
    return summary


def plot_results(test_scores, output):
    """Plot the single reserved-test comparison without reselecting a model."""
    subset = test_scores[test_scores.population.eq("All reserved test")]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(subset.role, subset.mae, color=cfg.CHART_COLOURS[0])
    ax.set(xlabel="Reserved-test MAE ($)", title="Frozen models on the same test rows")
    ax.invert_yaxis()
    fig.text(.02, .02, "Model recommendation was fixed using development CV before these scores were read.", fontsize=8)
    fig.tight_layout(rect=[0, .06, 1, 1])
    fig.savefig(output / "test_comparison.png", dpi=cfg.CHART_DPI)
    plt.close(fig)


def run_decisions(df, development, predictions, test_scores, selection, output: Path):
    """Export descriptive evidence, narrow group comparisons, and a review pilot."""
    output.mkdir(parents=True, exist_ok=True)
    queues = build_review_outputs(development, predictions, output)
    compare_groups(df, output, cfg.RANDOM_SEED)
    minimum = float(df.Mileage.min())
    mileage_flag = df.Mileage.eq(minimum)
    df[["Source_Record_Number", "Mileage"]].assign(Mileage_At_Recorded_Minimum=mileage_flag).to_csv(output / "mileage_review_flags.csv", index=False)
    write_json(output / "mileage_minimum_audit.json", {
        "observed_minimum": minimum, "records_at_minimum": int(mileage_flag.sum()),
        "median_age_at_minimum": float(df.loc[mileage_flag, "Vehicle_Age"].median()),
        "interpretation": "Repeated minimum observed; clipping unconfirmed. Audit flag only; not a predictor.",
    })
    selected_test = test_scores.loc[(test_scores.role == "selected") & (test_scores.population == "All reserved test")].iloc[0]
    decisions = pd.DataFrame([
        {"decision": "Pilot balanced listing review", "action": "Review selected and random records with the same evidence and blinded assignment",
         "measure": "Verified actionable findings per reviewer-hour, by price band; commercial outcomes after action", "missing_data": "Verified specifications/history, review time, realised outcomes"},
        {"decision": "Understand repeated minimum values", "action": "Ask the source owner about price and mileage conventions before applying policies",
         "measure": "Confirmed source explanation and verified corrections", "missing_data": "Data collection and preprocessing rules"},
        {"decision": "Determine purchasing priorities", "action": "Join acquisition, reconditioning, holding and sales records before changing stock mix",
         "measure": "Realised contribution, days to sell, and return on invested capital", "missing_data": "Costs, final sale price, dates, unsold stock and ownership"},
    ])
    decisions.to_csv(output / "decision_table.csv", index=False)
    lines = ["# Decision memo", "", f"Selected model: {selection['selected']['family']} ({selection['selected']['candidate_id']}).",
             f"Internally reserved-test MAE: ${selected_test.mae:,.2f}. This is a listed-price estimate, not a confirmed market value.", "",
             "The review sample uses development-only out-of-fold predictions. Parameters were selected using these folds: audit scores remain retrospective and optimistic as predictive performance estimates.",
             "The reserved test is not used to construct the review queue. Outcomes are blank because no analyst has verified these records.", "",
             "| Queue | Records | Premium makes | At recorded minimum |",
             "|---|---:|---:|---:|"]
    for row in queues.itertuples():
        lines.append(f"| {row.queue} | {row.n} | {row.premium_makes} | {row.recorded_minimum} |")
    lines += ["", "Use blinded_review_sheet.csv for the reviewer; keep review_assignment_key.csv with the evaluator. Band matching balances broad price mix only. Compare useful findings per reviewer-hour within bands; record inconclusive cases and model errors too.",
              "The per-band centring/scaling references are estimated on the same development predictions. Equal allocation ensures coverage; it does not prove that cases are mispriced.", "",
              "Comparable-group results are descriptive backup evidence. Service overlap can be narrow; SUV/sedan comparisons retain model mix. Bootstrap intervals cover only resampling within retained strata and fixed weights, not confounding or design uncertainty.", "",
              f"All-data quality finding: {int(mileage_flag.sum())} rows record mileage {minimum:g}. Confirm its origin; the flag is not used as a model input.",
              "Better price prediction alone cannot establish profit, demand, turnover, or a preferred purchasing mix. No such data were added by tuning.",
              "The new audit population and selected estimator differ from the earlier workspace. Its queue counts should not be presented as a direct controlled improvement over those earlier counts.", ""]
    (output / "DECISION_MEMO.md").write_text("\n".join(lines), encoding="utf-8")
    plot_results(test_scores, output)


def compare_groups(df: pd.DataFrame, output: Path, seed: int):
    """Transparent stratified comparisons; not causal treatment estimates.

    Keep strata with the configured minimum observations in each group. Weight each stratum by the
    smaller group size, giving both groups the SAME mix of strata. Bootstrap
    within retained strata/groups, holding weights fixed: intervals describe
    sampling variability conditional on this overlap, NOT confounding uncertainty.
    """
    definitions = cfg.MATCH_COMPARISONS
    summaries, details = [], []
    rng = np.random.default_rng(seed)
    for population, subset in [('All',df),('Above floor',df[df.Selling_Price>cfg.RECORDED_PRICE_MINIMUM])]:
        for age_width in cfg.MATCH_AGE_WIDTHS:
            for label,field,a,b,additional in definitions:
                eligible = subset[subset[field].isin([a,b])].copy()
                eligible['Age_Block'] = (eligible.Vehicle_Age//age_width).astype(int)
                eligible['Mileage_Block'] = (eligible.Mileage//cfg.MATCH_MILEAGE_WIDTH).astype(int)
                keys = additional+['Age_Block','Mileage_Block']
                strata = []
                for key,cell in eligible.groupby(keys,observed=True):
                    ga,gb = cell[cell[field]==a],cell[cell[field]==b]
                    if min(len(ga),len(gb)) < cfg.MATCH_MINIMUM_PER_GROUP:
                        continue
                    weight = min(len(ga),len(gb))
                    strata.append((ga,gb,weight))
                    details.append(dict(population=population,comparison=label,age_bin_width=age_width,
                        stratum=str(key),n_a=len(ga),n_b=len(gb),weight=weight,
                        mean_price_a=ga.Selling_Price.mean(),mean_price_b=gb.Selling_Price.mean()))
                na,nb = int((eligible[field]==a).sum()),int((eligible[field]==b).sum())
                base = dict(population=population,comparison=label,group_a=a,group_b=b,
                    age_bin_width=age_width,mileage_bin_width=cfg.MATCH_MILEAGE_WIDTH,eligible_n_a=na,eligible_n_b=nb,
                    raw_mean_a=eligible.loc[eligible[field]==a,'Selling_Price'].mean(),
                    raw_mean_b=eligible.loc[eligible[field]==b,'Selling_Price'].mean(),
                    retained_strata=len(strata))
                if not strata:
                    summaries.append({**base,'status':'Insufficient overlap'})
                    continue
                total_weight = sum(w for _,_,w in strata)
                totals = {c:[0.,0.] for c in ['Selling_Price','Vehicle_Age','Mileage']}
                boot = np.zeros(cfg.BOOTSTRAP_REPLICATES)
                for ga,gb,w in strata:
                    for c in totals:
                        totals[c][0] += w*ga[c].mean()/total_weight
                        totals[c][1] += w*gb[c].mean()/total_weight
                    va,vb = ga.Selling_Price.to_numpy(),gb.Selling_Price.to_numpy()
                    sampled_a = rng.choice(va,size=(cfg.BOOTSTRAP_REPLICATES,len(va)),replace=True).mean(axis=1)
                    sampled_b = rng.choice(vb,size=(cfg.BOOTSTRAP_REPLICATES,len(vb)),replace=True).mean(axis=1)
                    boot += w*(sampled_a-sampled_b)/total_weight
                retained_a = sum(len(a) for a,_,_ in strata)
                retained_b = sum(len(b) for _,b,_ in strata)
                pa,pb = totals['Selling_Price']
                summaries.append({**base,'status':'Exploratory overlap comparison',
                    'retained_n_a':retained_a,'retained_n_b':retained_b,
                    'coverage_a_pct':100*retained_a/na,'coverage_b_pct':100*retained_b/nb,
                    'standardised_mean_a':pa,'standardised_mean_b':pb,
                    'difference_a_minus_b':pa-pb,'difference_pct_of_b':100*(pa-pb)/pb,
                    'conditional_bootstrap_low':np.quantile(boot,(1-cfg.CONFIDENCE_LEVEL)/2),
                    'conditional_bootstrap_high':np.quantile(boot,(1+cfg.CONFIDENCE_LEVEL)/2),
                    'weighted_age_a':totals['Vehicle_Age'][0],'weighted_age_b':totals['Vehicle_Age'][1],
                    'weighted_mileage_a':totals['Mileage'][0],'weighted_mileage_b':totals['Mileage'][1]})
    result = pd.DataFrame(summaries)
    result.to_csv(output/'comparable_group_summary.csv',index=False)
    pd.DataFrame(details).to_csv(output/'comparable_group_strata.csv',index=False)
    return result
