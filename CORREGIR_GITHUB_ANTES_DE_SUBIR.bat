@echo off
cd /d "%~dp0"

echo.
echo ============================================================
echo FOOTBALL EDGE PRO - LIMPIAR GITHUB ANTES DEL PRIMER COMMIT
echo ============================================================
echo.

git --version
if errorlevel 1 (
  echo ERROR: Git no esta disponible.
  pause
  exit /b 1
)

echo [1/4] Quitando todo del staging anterior...
git reset

echo.
echo [2/4] Aplicando .gitignore seguro...
echo     - .env NO se subira
echo     - secrets.toml NO se subira
echo     - caches raw de FotMob NO se subiran
echo     - data/*.db y models/*.joblib SI se mantienen

echo.
echo [3/4] Volviendo a preparar solo los archivos correctos...
git add .

echo.
echo [4/4] Comprobacion final...
echo.
git status --short

echo.
echo ============================================================
echo REVISA ARRIBA:
echo NO debe aparecer ".env"
echo NO debe aparecer "data/fotmob_referee_raw/"
echo NO debe aparecer ".streamlit/secrets.toml"
echo.
echo SI deben aparecer streamlit_app.py, requirements.txt,
echo data/*.db y models/*.joblib
echo ============================================================
echo.
pause
