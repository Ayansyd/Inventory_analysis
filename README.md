The analysis reads workspace/data/dataset.csv and writes generated files to workspace/results/. These results are excluded from Git and can be recreated by running the commands above. Tuning may take several minutes, depending on your computer.
What the analysis does
- Keeps incomplete rows and reports missing information rather than silently removing or filling values.
- Summarises relationships between vehicle details and listed prices.
- Compares model performance using five-fold cross-validation and evaluates the selected configuration on a reserved portion of the data.
- Exports charts, tables, tuning results and management summaries.
The selected model achieved a $1,417 MAE on the reserved listings, compared with $2,246 for the fixed forest reference. This is an internal evaluation: the dataset had already been explored, so it does not establish performance on future listings.
Important limitation
Prices are listing asking prices, not confirmed sale prices. The dataset does not include acquisition or repair costs, final sale prices, or time to sale, so it cannot establish profit, demand, or an optimal purchasing mix.
Project files
- workspace/config.py — analysis settings and tuning ranges
- workspace/clean_data.py — data validation and cleaning
- workspace/run_analysis.py — runs the analysis workflow
- workspace/tune_models.py — model search and evaluation
- workspace/Analysis.py — descriptive analysis and staged model comparisons
- workspace/Decision_analysis.py — comparison and review-selection analyses
- workspace/management_evidence.py — exports management-focused findings
- workspace/requirements.txt — Python dependencies
- presentations/ — technical and management presentation decks
