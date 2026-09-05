@echo off
cd /d "%~dp0"

echo.
echo ============================================================
echo FOOTBALL EDGE PRO - LIMPIEZA GITHUB V2
echo ============================================================
echo.

git --version
if errorlevel 1 (
  echo ERROR: Git no esta disponible.
  pause
  exit /b 1
)

echo [1/5] Quitando staging anterior...
git reset

echo.
echo [2/5] Aplicando .gitignore V2...
echo     - .env y secrets reales fuera
echo     - cache data/laliga_raw/fotmob_referees/ fuera
echo     - caches FotMob fuera
echo     - data/*.db y models/*.joblib se mantienen

echo.
echo [3/5] Preparando de nuevo...
git add .

echo.
echo [4/5] Verificaciones automaticas...
echo.
echo --- Archivos sensibles/caches que NO deben salir ---
git status --short | findstr /I /R /C:"^A  \.env$" /C:"secrets.toml$" /C:"data/laliga_raw/fotmob_referees/" /C:"data/fotmob_referee_raw/"
if errorlevel 1 (
  echo OK - no hay secretos ni caches masivas en staging.
) else (
  echo ERROR - todavia hay archivos que no deben subirse.
  pause
  exit /b 1
)

echo.
echo --- Archivos esenciales que SI deben salir ---
git status --short | findstr /I /C:"streamlit_app.py" /C:"requirements.txt" /C:"data/laliga.db" /C:"data/eliteserien.db" /C:"models/laliga_goals_v4_team_strength_poisson.joblib"

echo.
echo [5/5] Resumen corto...
echo.
git status --short

echo.
echo ============================================================
echo SI ARRIBA PONE:
echo   OK - no hay secretos ni caches masivas en staging.
echo entonces ya estamos listos para hacer el primer commit.
echo ============================================================
echo.
pause
