# HPNow — Faradaic Efficiency Degradation Analyser

A Streamlit web app for analysing Faradaic efficiency degradation across electrolyser test runs. Experiment data is stored in Supabase (PostgreSQL) and a Gemini-powered AI assistant answers analytical questions about the dataset.

**Primary question:** what operating conditions or design choices are associated with faster Faradaic efficiency degradation over time?

---

## Features

| Tab | Description |
|-----|-------------|
| **📉 Degradation Overview** | Per-run degradation rates (% efficiency lost per 100 hours) via linear regression, sortable table, Excel export |
| **📈 Efficiency Trajectories** | Interactive time-series plot — hover a run line to see metadata, click to pre-populate Run Detail |
| **🔍 Correlations** | Pearson correlations between measured variables and Faradaic efficiency; box plots per GDL / project / operator |
| **🏭 Station Comparison** | Cross-station median degradation rates and run counts |
| **💬 Ask AI** | Multi-turn Gemini assistant with live dataset access |
| **➕ Migrate Run** | Migrate a completed experiment from the live Google Sheet into Supabase |
| **🔬 Run Detail** | Full per-run view: metadata card, efficiency trajectory with trend line, supporting measurement charts, notes and metadata editing |

**Sidebar filters** narrow the dataset by test station, project, year, minimum run duration, and whether to include informal runs.

---

## Architecture

### Data flow

```
Live Google Sheet  ──► (engineer runs Migrate Run tab)──► Supabase (PostgreSQL)
                                                               │
                                                          fetch_db.py
                                                               │
                                                           app.py (Streamlit)
```

Completed experiments live in Supabase. When an experiment finishes, the engineer uses the **➕ Migrate Run** tab (or `migrate_run.py` CLI) to move it from the live Google Sheet into Supabase, then clears that slot in the sheet for the next experiment.

### Key files

```
app.py                      Streamlit UI — all tabs, plots, AI chat
fetch_db.py                 Reads all runs and measurements from Supabase
fetch_sheets.py             Parses Google Sheet tabs (used during migration)
supabase_utils.py           Shared DB utilities: column mapping, insert_run(), get_client()
config.py                   Sheet IDs, credentials path, analysis thresholds
migrate_run.py              CLI: migrate one run from the live sheet into Supabase
migrate_historical.py       CLI: one-time bulk migration from the storage sheet
.github/workflows/
  supabase_keep_alive.yml   Pings Supabase every 3 days to prevent free-tier pause
  supabase_backup.yml       Weekly CSV backup of runs + measurements tables
.github/scripts/
  backup_supabase.py        Backup script called by the backup workflow
backups/
  runs.csv                  Latest weekly snapshot of the runs table
  measurements.csv          Latest weekly snapshot of the measurements table
```

### Database schema (Supabase)

Two tables in PostgreSQL:

**`runs`** — one row per experiment
| Column | Type | Description |
|--------|------|-------------|
| `id` | uuid | Primary key |
| `source_key` | text | Deduplication key (`formal::stack_id::date_start` or positional fallback) |
| `tab_name` | text | Test station (e.g. `r0052`) |
| `stack_id` | text | Electrolyser stack identifier |
| `operator` | text | Engineer initials |
| `date_start` | text | Experiment start date/time |
| `project` | text | Project name |
| `gdl` | text | Gas diffusion layer type |
| `n_cells` | integer | Number of cells in stack |
| `cell_area_cm2` | numeric | Cell active area |
| `current_ma_cm2` | numeric | Applied current density |
| `notes` | text | Free-text engineer annotations (added in app) |
| `is_informal` | boolean | True if no formal metadata block |
| `migrated_at` | timestamptz | When the run was migrated |

**`measurements`** — one row per logged data point, foreign-keyed to `runs`

---

## Local development

### Prerequisites

- Python 3.11+
- Supabase service account key
- GCP service account JSON with read access to the Google Sheets
- Google AI Studio API key (free — for the AI chat tab)

### Setup

```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "your-gemini-api-key"

[supabase]
url         = "https://<project-id>.supabase.co"
service_key = "eyJ..."

[gcp_service_account]
type                        = "service_account"
project_id                  = "..."
private_key_id              = "..."
private_key                 = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email                = "...@....iam.gserviceaccount.com"
client_id                   = "..."
auth_uri                    = "https://accounts.google.com/o/oauth2/auth"
token_uri                   = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url        = "..."
universe_domain             = "googleapis.com"
```

Alternatively, place `service_account.json` in the project root for local Google Sheets access.

### Run

```bash
streamlit run app.py
```

---

## Migrating a completed run

When an experiment finishes:

1. Open the app → **➕ Migrate Run** tab
2. Select the station tab from the live sheet
3. Review the metadata preview
4. Click **Migrate to database**
5. Once the green confirmation appears, clear that slot in the live sheet

The migration is idempotent — running it again on the same run updates measurements but preserves any notes or metadata edits made in the app.

### CLI alternative

```bash
python migrate_run.py --tab r0052
```

---

## Run annotations and metadata editing

In the **🔬 Run Detail** tab, select any station and run to:

- **Add notes** — free-text observations saved to `runs.notes` in Supabase (never overwritten by re-migration)
- **Edit metadata** — correct stack ID, operator, GDL, project, aim, or numeric fields directly from the app

---

## Degradation classification

Each run is classified by the linear regression slope of Efficiency (%) vs Time (hours):

| Label | Condition |
|-------|-----------|
| `degrading` | slope < −0.03 %/h (worse than −3 %/100 h) |
| `stable` | −0.03 ≤ slope ≤ +0.01 %/h |
| `improving` | slope > +0.01 %/h |
| `short` | run duration < 20 hours |
| `inconclusive` | insufficient data for regression |
| `no_data` | fewer than 4 valid data points |

---

## Deployment (Streamlit Community Cloud)

The app deploys automatically on every push to `main`.

### Secrets in Streamlit Cloud

Configure these in the Streamlit Cloud dashboard under **App settings → Secrets**:

| Secret | Purpose |
|--------|---------|
| `GOOGLE_API_KEY` | Gemini API key for the AI assistant |
| `[gcp_service_account]` | GCP service account for Google Sheets read access |
| `[supabase] url / service_key` | Supabase connection for all data reads and writes |

### GitHub Actions secrets

| Secret | Used by |
|--------|---------|
| `SUPABASE_KEY` | Keep-alive ping + weekly backup workflows |

---

## Automated workflows (GitHub Actions)

### Keep Supabase alive
**File:** `.github/workflows/supabase_keep_alive.yml`  
Runs every 3 days. Pings the Supabase REST API to prevent the free-tier 7-day inactivity pause.

### Weekly backup
**File:** `.github/workflows/supabase_backup.yml`  
Runs every Sunday at 02:00 UTC. Exports both `runs` and `measurements` tables to `backups/runs.csv` and `backups/measurements.csv` and commits them to the repo. Only creates a commit if data has changed.

**Restoring from backup:**
```python
import pandas as pd
runs = pd.read_csv("backups/runs.csv")
meas = pd.read_csv("backups/measurements.csv")
```

Git history gives point-in-time recovery — any previous weekly snapshot is accessible via:
```bash
git show HEAD~4:backups/runs.csv
```

Both workflows can be triggered manually from the **Actions** tab at any time.

---

## Security

- `service_account.json` and `.app_config.json` are excluded from git via `.gitignore`
- All credentials are stored exclusively in Streamlit's encrypted secrets management and GitHub Actions secrets — never in the repository
- The Supabase service key has full database access; rotate it via the Supabase dashboard if compromised
- App access is restricted to whitelisted emails via Streamlit viewer authentication
