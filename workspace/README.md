# Run the analysis

Use Python **3.10+**. From this extracted folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_analysis.py
python management_evidence.py
```

Windows activation: `.venv\Scripts\activate`. The supplied `results/` folder is empty; running the command creates the results. No API keys are needed. Tuning may take several minutes or longer depending on your CPU.

For a separate run: `python run_analysis.py --output results_new`, then `python management_evidence.py --results results_new`. Existing results are never overwritten. To use another raw dataset: add `--input path/to/dataset.csv` with the same column schema.

## Change settings in one place

Edit **`config.py`** before running. It contains cleaning assumptions, predictors, split/seed/folds, all search values, fixed reference parameters, comparison bins, bootstrap settings, review quotas, and chart settings. The input dataset is included in `data/`.

## What tuning does

1. Reserve about 20% of description groups. Identical recorded descriptions stay together; this does not replace a VIN.
2. Use five common folds on the remaining data to try **48 Ridge scenarios + 36 Random Forest combinations + one median baseline = 85 candidates (425 CV fits)**.
3. Minimise validation **MAE**, the average dollar miss. Record all fold and training errors. Unbounded Ridge is diagnostic; recommendation candidates respect nonnegative prices here.
4. Record the lowest-error candidate. Recommend the simplest candidate within one approximate standard error of it, using a fixed rule declared in code. This does not prove statistical equality or global optimality.
5. Freeze the recommendation; evaluate it, the CV winner, and fixed reference models on the same reserved test. Stop without further search. The saved model is fitted on development rows only.

| Model | Parameter values tried |
|---|---|
| Ridge | `alpha`: 0.1, 1, 10, 100; spline knots: 3, 4, 6; degree: 2, 3; minimum prediction: none or $500 |
| Random Forest | trees: 250, 500; maximum depth: 8, 16, unlimited; minimum leaf records: 2, 5, 10; feature fraction: 0.7, 1.0 |

These ranges compare weak/strong regularisation, simpler/more flexible curves, and shallower/deeper forests while keeping a finite budget. More trees tests stability; depth and leaf size limit how closely trees can memorise small groups. A $500 constraint remains a dataset assumption.

## Read these after running

- `results/tuning/TUNING_REPORT.md`: tested ranges, best and recommended parameters, why selected, final test results, and limits.
- `results/tuning/search_results.csv`: every combination, each fold's error, training error, timing, and selection flags. `search_plan.json` is the full predeclared list.
- `results/tuning/selected_parameters.json`: the frozen recommendation and CV winner.
- `results/decision/DECISION_MEMO.md`: the review proposal and business boundaries; reviewer outcome fields are blank.
- `results/exploratory/`: chart images and their evidence tables.
- `results/management/`: all-listing age/value shares, make/body-type descriptive ratios, and current audit counts for revising the management deck.

Exact package versions, input/code hashes, the config snapshot, and record-to-split assignments are saved for reproducibility. The reference experiment may use newer packages than your Python supports; version differences can slightly change results.

## Files

`run_analysis.py` coordinates; `clean_data.py` validates and derives columns; `tune_models.py` splits, searches and evaluates; `Analysis.py` creates descriptions and fixed staged comparisons; `Decision_analysis.py` builds the review audit; `management_evidence.py` exports management figures; `predict.py` applies a saved model. All settings are in `config.py`.

Optional prediction on a complete-schema raw listing CSV (load only your own saved model):

```bash
python predict.py --model results/tuning/selected_model.joblib --input data/dataset.csv --output listing_estimates.csv
```

Those predictions are for inspection, not a new accuracy test on training data. Current listed price is required by the input schema but is excluded from the estimator inputs.

## Interpretation

This dataset has already been explored: the reserved partition is protected from this new search, not from all prior human knowledge. Fresh dated listings are needed for independent validation. Repeatedly inspecting test scores and changing settings contaminates the test; changing the seed does not fix that.

Use this run's reference models to measure tuning gains. Old full-dataset CV scores used a different evaluation design. Better price prediction cannot establish profit or an optimal stock mix without costs, realised sale prices, and sales dates. Audit usefulness still requires human verification.
