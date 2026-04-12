SHEET_ID = "1xbZr7x9mbtTvXfaiOj1oHAUa1Py5THdRKHgqMVerPNE"

CREDENTIALS_FILE = "service_account.json"  # Service account key downloaded from GCP

# Tabs to skip (add sheet names here if any are non-data tabs)
SKIP_TABS = []

# The target variable for all correlation analysis
TARGET_COL = "Efficiency (%)"

# Minimum number of data rows for a tab to be included
MIN_ROWS = 3

# A run is "degrading" if efficiency drops by more than this percentage
# over the course of the test
DEGRADATION_THRESHOLD_PCT = 10.0

# A run is "long-stable" if it lasts this many hours with <DEGRADATION_THRESHOLD_PCT drop
STABLE_MIN_HOURS = 200
