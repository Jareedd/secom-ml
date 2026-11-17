"""Project-wide settings.

Every choice that could change a result lives here, so there is one place to
look when someone asks "what did you assume?".
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"

RANDOM_SEED = 42

# Stratified cross-validation folds, used on the training runs only.
N_FOLDS = 5

# Time-ordered check: the training runs are cut into 5 consecutive blocks, and
# each of the last 4 is scored by a model fit on all the blocks before it.
N_TIME_FOLDS = 4

RESULTS_DIR = PROJECT_ROOT / "results"

# Cost assumptions. These are assumptions, not measurements: SECOM comes with
# no cost data. Costs are in units of one false-alarm inspection.
COST_FALSE_ALARM = 1.0
# How many times worse a missed failing run is than a false alarm. Every result
# is reported for each of these, because the right ratio is a business question.
COST_RATIOS = [5, 10, 20, 50]

# The one ratio quoted in the headline result. Fixed before the test set was
# opened, so it could not be picked for looking best. The headline saving is
# measured against the cheaper of "never alarm" and "inspect every run".
HEADLINE_COST_RATIO = 10

# Bootstrap resamples behind every confidence interval.
N_BOOTSTRAP = 2000

# The most recent 20% of runs are the final test set. Nothing may be decided
# by looking at them; they are scored exactly once, at the end.
TEST_FRACTION = 0.20

# Cleaning rule, decided on training runs only: a sensor is dropped when more
# than this share of its readings are missing. (Sensors that never change are
# dropped too; that rule has no number to set.)
MAX_MISSING_SHARE = 0.50
