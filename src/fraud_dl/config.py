"""Settings shared by the data pipeline, the baseline and later models."""

SECONDS_PER_DAY = 86_400

# Time split, counted back from the last day in train_transaction.csv (about 182 days).
TEST_DAYS = 30
VAL_DAYS = 30
# Days dropped between train and val/test. Real chargebacks arrive weeks late, so a
# gap is more honest; 0 keeps the most data. Changing it changes every result row.
GAP_DAYS = 0

# Velocity windows in seconds.
VELOCITY_WINDOWS = {"1h": 3_600, "24h": SECONDS_PER_DAY, "7d": 7 * SECONDS_PER_DAY}

# Transformer inputs (week 5). The label is never an input.
SEQ_LEN = 32
MIN_CAT_COUNT = 5  # categories seen fewer times in train become UNK
PAD_ID, UNK_ID = 0, 1

SEQ_CAT_COLS = [
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType", "DeviceInfo", "id_30", "id_31", "id_33",
]

# Numeric inputs: log1p is applied to the non-negative, heavy-tailed ones, then each is
# standardised with train statistics and missing values become 0 (the train mean).
SEQ_LOG_COLS = ["TransactionAmt", "secs_since_prev", "dist1",
                "C1", "C2", "C5", "C6", "C13", "C14"]
SEQ_PLAIN_COLS = ["hour_sin", "hour_cos", "D1", "D4", "D10", "D15"]

# Never fed to any model: identifiers, raw time and the label.
NON_FEATURE_COLS = ["TransactionID", "TransactionDT", "isFraud", "day", "uid", "split"]
