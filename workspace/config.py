"""Edit experiment settings HERE, before running the program.

Keep the search budget and split fixed once results have been inspected.
Changing settings after seeing the test score turns that test into development
data. A new random seed does not create genuinely new evidence.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "data" / "dataset.csv"
OUTPUT_DIRECTORY = ROOT / "results"

# Data assumptions, not claims about when the cars were actually listed.
REFERENCE_YEAR = 2024
RECORDED_PRICE_MINIMUM = 500.0
EARLIEST_VEHICLE_YEAR = 1886
RANDOM_SEED = 42
TEST_FRACTION = 0.20
CV_FOLDS = 5
NUMERICAL_THREADS = 1
FOREST_WORKERS = 2  # Parallelise trees; candidate searches run sequentially.

TARGET = "Selling_Price"
NUMERIC_COLUMNS = ["Year", "Engine_Size", "Mileage", "Horsepower", "Torque",
                   "Owners", "Accident_History", "Fuel_Efficiency", TARGET]
TEXT_COLUMNS = ["Make", "Model", "Fuel_Type", "Transmission", "Service_History",
                "Color", "Body_Type", "Drivetrain", "Location"]
REQUIRED_COLUMNS = NUMERIC_COLUMNS + TEXT_COLUMNS
CORE_COLUMNS = ["Make", "Model", "Year", "Fuel_Type", "Mileage", "Owners",
                "Body_Type", "Drivetrain", TARGET]
UNKNOWN_COLUMNS = ["Transmission", "Service_History", "Color", "Location"]
DERIVED_COLUMNS = ["Vehicle_Age", "At_Price_Floor", "Accident_Status",
                   "Efficiency_Requires_Separate_Analysis"]
ALLOWED_CATEGORIES = {
    "Fuel_Type": {"Petrol", "Diesel", "Hybrid", "Electric"},
    "Transmission": {"Automatic", "Manual"},
    "Service_History": {"Full Service", "Partial Service", "No Service"},
    "Body_Type": {"SUV", "Sedan", "Truck", "Coupe", "Hatchback"},
    "Drivetrain": {"FWD", "AWD", "RWD", "4WD"},
}
AGE_MILEAGE = ["Vehicle_Age", "Mileage"]
CONDITION_FEATURES = ["Owners", "Accident_Status", "Service_History"]
OTHER_FEATURES = ["Fuel_Type", "Body_Type", "Drivetrain", "Transmission", "Location"]
FULL_FEATURES = AGE_MILEAGE + ["Model"] + CONDITION_FEATURES + OTHER_FEATURES
NUMERIC_FEATURES = AGE_MILEAGE + ["Owners"]
# Keep identical recorded vehicle descriptions together in both splits and CV.
# These columns exclude price and all target-derived flags. They are NOT a VIN.
DUPLICATE_GROUP_COLUMNS = [c for c in REQUIRED_COLUMNS if c != TARGET]

# Fixed comparisons from the earlier approach. Both are also in the search grid.
RIDGE_REFERENCE = {"alpha": 10.0, "n_knots": 4, "degree": 3,
                   "prediction_minimum": RECORDED_PRICE_MINIMUM}
STAGED_RIDGE = {**RIDGE_REFERENCE, "prediction_minimum": None}
FOREST_REFERENCE = {"n_estimators": 250, "max_depth": 16,
                    "min_samples_leaf": 5, "max_features": 1.0}
RIDGE_FIXED = {"solver": "lsqr", "tol": 1e-6}
SPLINE_FIXED = {"include_bias": False, "extrapolation": "linear"}
ENCODER_SETTINGS = {"handle_unknown": "ignore", "sparse_output": False}
FOREST_FIXED = {"criterion": "squared_error", "bootstrap": True,
                "min_samples_split": 2, "random_state": RANDOM_SEED,
                "n_jobs": FOREST_WORKERS}

# 4 x 3 x 2 x 2 = 48 Ridge scenarios. Unconstrained predictions are retained
# for diagnostics; only the constrained scenario can be recommended here.
RIDGE_GRID = {"alpha": [0.1, 1.0, 10.0, 100.0], "n_knots": [3, 4, 6],
              "degree": [2, 3], "prediction_minimum": [None, RECORDED_PRICE_MINIMUM]}
# 2 x 3 x 3 x 2 = 36 Random Forest combinations. No expansion after the run.
FOREST_GRID = {"n_estimators": [250, 500], "max_depth": [8, 16, None],
               "min_samples_leaf": [2, 5, 10], "max_features": [0.7, 1.0]}
PRIMARY_METRIC = "MAE"  # Dollars: minimise the average absolute prediction miss.
SELECTION_STANDARD_ERRORS = 1.0
# Prefer the simpler model among candidates within this many standard errors of
# the best CV MAE. The standard error is a heuristic: CV folds are not independent.
ERROR_PERCENTILE = 0.90
SAVE_MODEL_COMPRESSION = 3

# Descriptive analysis and audit parameters; none are pricing-model predictors.
SMALL_GROUP_N = 30
CONFIDENCE_LEVEL = 0.95
AGE_EDGES = [-1, 4, 9, 14, float("inf")]
AGE_LABELS = ["0-4", "5-9", "10-14", "15+"]
MILEAGE_EDGES = [0, 50000, 100000, 150000, 200000, float("inf")]
MILEAGE_LABELS = ["<50k", "50k-<100k", "100k-<150k", "150k-<200k", "200k+"]
PRICE_EDGES = [0, RECORDED_PRICE_MINIMUM, 5000, 15000, 30000, float("inf")]
PRICE_LABELS = ["At/below recorded minimum", ">minimum-5k", ">5k-15k", ">15k-30k", ">30k"]
SEGMENTS = ["Make", "Model", "Body_Type", "Fuel_Type", "Drivetrain", "Location",
            "Accident_Status", "Service_History", "Owners", "Transmission"]
CORRELATION_COLUMNS = ["Vehicle_Age", "Mileage", "Owners", "Engine_Size", "Horsepower", "Torque"]
CORRELATION_MINIMUM_N = 3
PREMIUM_MAKES = ["Audi", "BMW", "Mercedes-Benz"]
REVIEW_PER_PRICE_BAND = 10
MATCH_AGE_WIDTHS = [3, 5]
MATCH_MILEAGE_WIDTH = 50000
MATCH_MINIMUM_PER_GROUP = 3
BOOTSTRAP_REPLICATES = 600
MATCH_COMPARISONS = [
    ("Accident: Yes vs No", "Accident_Status", "Yes", "No", ["Model"]),
    ("Service: Full vs No", "Service_History", "Full Service", "No Service", ["Model", "Accident_Status"]),
    ("Body: SUV vs Sedan", "Body_Type", "SUV", "Sedan", ["Fuel_Type"]),
]
CHART_DPI = 160
HISTOGRAM_BINS = 45
HEXBIN_CELLS = 32
CHART_COLOURS = ["#246A90", "#D57B37", "#657583"]
