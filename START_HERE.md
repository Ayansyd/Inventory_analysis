# Run the project

Open workspace/README.md for installation and run commands. Python 3.10+ and requirements.txt are required. All adjustable analysis and tuning settings are in workspace/config.py. 
## Contents

- workspace/: complete commented source code, requirements and supplied raw dataset.
- presentations/technical_review_tuned.pptx: six main slides plus two hidden backup slides.
- presentations/management_review_tuned.pptx: five main slides plus one hidden backup slide.
- reference_results/: completed experiment results, charts, all tested parameter combinations, saved model and review sheets.

## Read only these first

1. workspace/README.md for running and file roles.
2. reference_results/tuning/TUNING_REPORT.md for tested settings, selection and evaluation.
3. reference_results/decision/DECISION_MEMO.md for the human-review pilot and limits.

## Final analytical position

The bounded search evaluated 85 candidates across five development folds. The selected forest has 250 trees, max_depth=None, min_samples_leaf=2 and max_features=1.0. Its reserved-set MAE is $1,417 against $2,246 for the fixed same-run forest reference. The 500-tree option only improved CV MAE by about $4, so the predefined simplicity rule chose 250. No post-test tuning followed.

The current audit has 40 premium cars in the dollar-gap queue and 36 minimum-price cars in the percentage-gap queue. The balanced sample and controls each have 50 records, ten per price band. Premium counts still differ: 28 versus 19. Human review has not established usefulness or profit.

Previously explored data limits the independence of this internal reservation. Fresh dated listings, vehicle IDs, costs and realised sales are needed for future-market validation and commercial decisions.

The earlier $34 model-gap claim and older audit counts have been replaced in these decks. Body-type percentages describe older-to-younger median asking-price ratios, not inventory shares. No files have been emailed or submitted.
