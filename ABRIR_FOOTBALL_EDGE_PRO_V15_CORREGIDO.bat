@echo off
cd /d "%~dp0"

echo.
echo ============================================================
echo FOOTBALL EDGE PRO V15 - ABRIENDO APP
echo ============================================================
echo.

start "" cmd /c "timeout /t 3 /nobreak >nul & start "" http://localhost:8501"

python -m streamlit run app_v15_analista_visual.py --server.port 8501

pause
