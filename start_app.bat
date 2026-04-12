@echo off
cd /d "%~dp0"
echo Starting HPNow Efficiency Analyser...
echo.
echo The app will open in your browser at http://localhost:8501
echo Close this window to stop the app.
echo.
python -m streamlit run app.py
pause
