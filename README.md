# HPNow — Faradaic Efficiency Degradation Analyser

A Streamlit web app for analysing Faradaic efficiency degradation across electrolyser test runs. Data is read live from Google Sheets, and a Gemini-powered AI assistant answers analytical questions about the dataset.

**Primary question:** what operating conditions or design choices are associated with faster Faradaic efficiency degradation over time?

---

## Features

- **Degradation Overview** — per-run degradation rates (% efficiency lost per 100 hours) via linear regression, with sortable table and Excel export
- **Efficiency Trajectories** — time-series plots for any selection of runs, colour-coded by degradation rate or label
- **Correlations** — Pearson correlations between all measured variables and Faradaic efficiency, scatter plots for top features
- **Station Comparison** — cross-station bar charts comparing median degradation rates and run counts
- **Ask AI** — multi-turn chat assistant (Gemini) with full access to the filtered dataset; can run analytical queries and explain findings

**Sidebar filters** let you narrow by test station, project, minimum run duration, and whether to include informal runs (no stack ID metadata).

---

## Architecture

```
app.py              Streamlit app — all UI, plotting, and AI chat
fetch_sheets.py     Reads and parses all tabs from the Google Sheet
config.py           Sheet ID, credentials path, thresholds
analyze.py          Standalone analysis utilities
report.py           Report generation helpers
run_analysis.py     CLI entry point for offline analysis
```

Data flow:
1. `fetch_sheets.py` authenticates with a GCP service account and reads all worksheet tabs
2. Each tab is parsed into per-run DataFrames (handles formal metadata blocks and informal "New experiment" boundaries)
3. `app.py` computes degradation statistics (linear regression slope per run) and renders the UI

---

## Local Development

### Prerequisites

- Python 3.11
- A GCP service account JSON file with read access to the Google Sheet
- A [Google AI Studio](https://aistudio.google.com) API key (free)

### Setup

```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` (copy from `.streamlit/secrets.toml.example` and fill in real values):

```toml
GOOGLE_API_KEY = "your-gemini-api-key"

[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "...@....iam.gserviceaccount.com"
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
universe_domain = "googleapis.com"
```

Alternatively, place `service_account.json` in the project root (path configured in `config.py`) and enter the Gemini API key in the app sidebar at runtime.

### Run

```bash
streamlit run app.py
```

---

## Deployment (Streamlit Community Cloud)

The app is deployed at Streamlit Community Cloud with viewer authentication — only whitelisted email addresses can access it.

### To redeploy after code changes

```bash
git add .
git commit -m "your message"
git push
```

Streamlit automatically redeploys on every push to `main`.

### Initial deployment steps (for reference)

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **Create app** → **Deploy a public app from GitHub**
3. Set repository `HPNow/hpnow-efficiency-analyser`, branch `main`, main file `app.py`
4. Under **Advanced settings → Secrets**, paste the contents of `secrets.toml` (see template above)
5. Select **Python 3.11** and click **Deploy**
6. After deployment: app **Settings → Sharing** → restrict to specific email addresses

### Secrets managed in Streamlit Cloud

The following secrets are configured in the Streamlit Cloud dashboard (never in the repo):

| Secret | Purpose |
|---|---|
| `GOOGLE_API_KEY` | Gemini API key for the AI chat assistant |
| `[gcp_service_account]` | GCP service account for Google Sheets read access |

---

## Degradation Classification

Each run is classified based on the linear regression slope of Efficiency (%) vs Time (hours):

| Label | Condition |
|---|---|
| `degrading` | slope < −0.03 %/h (worse than −3 %/100 h) |
| `stable` | −0.03 ≤ slope ≤ +0.01 %/h |
| `improving` | slope > +0.01 %/h |
| `short` | run duration < 20 hours |
| `inconclusive` | insufficient data for regression |
| `no_data` | fewer than 4 valid data points |

---

## Security

- `service_account.json`, `credentials.json`, and `.app_config.json` are excluded from git via `.gitignore`
- All secrets are stored exclusively in Streamlit's encrypted secrets management
- The GitHub repository is public (source code only — no credentials)
- App access is restricted to whitelisted emails via Streamlit viewer authentication
