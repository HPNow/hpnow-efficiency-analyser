@echo off
cd /d "%~dp0"

:: ── DEV environment ─────────────────────────────────────────────────────────
:: These env vars override .streamlit/secrets.toml so the app talks to the
:: dev Supabase project instead of production.
::
:: 1. Go to https://supabase.com → New project → name it "hpnow-dev"
:: 2. Run schema.sql in the dev project's SQL editor
:: 3. Copy your dev project URL and service_role key below
:: 4. (Optional) run:  python seed_dev_db.py --runs 15
::    to copy a sample of prod runs into the dev project for testing

set SUPABASE_URL=PASTE_YOUR_DEV_PROJECT_URL_HERE
set SUPABASE_KEY=PASTE_YOUR_DEV_SERVICE_ROLE_KEY_HERE
set HPNOW_ENV=dev

:: ── Launch on a separate port so prod and dev can run side-by-side ───────────
echo.
echo  HPNow DEV  ^|  http://localhost:8502
echo  Supabase  ^|  %SUPABASE_URL%
echo.
echo  Keep this window open.  Press Ctrl+C to stop.
echo.

python -m streamlit run app.py --server.port 8502 --server.headless true
pause
