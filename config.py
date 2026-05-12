SHEET_ID = "1xbZr7x9mbtTvXfaiOj1oHAUa1Py5THdRKHgqMVerPNE"

# ID of the live Google Sheet where active experiments are tracked.
# Set this to the sheet ID from its URL: .../spreadsheets/d/<ID>/edit
LIVE_SHEET_ID = ""

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
