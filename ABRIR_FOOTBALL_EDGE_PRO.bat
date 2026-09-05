@echo off
title Football Edge Pro
cd /d "%~dp0"

echo ============================================================
echo FOOTBALL EDGE PRO - ACTUALIZANDO LALIGA
echo ============================================================
python update_laliga_fixtures_v2.py

if errorlevel 1 (
    echo.
    echo ERROR actualizando LaLiga.
    echo La app se abrira igualmente con los datos disponibles.
    echo.
    pause
)

echo.
echo ============================================================
echo ABRIENDO FOOTBALL EDGE PRO
echo ============================================================
streamlit run app_v6_multiliga.py
