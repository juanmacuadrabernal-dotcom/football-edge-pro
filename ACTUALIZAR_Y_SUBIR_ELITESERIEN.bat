@echo off
setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo FOOTBALL EDGE PRO - ACTUALIZAR ELITESERIEN + GITHUB
echo ============================================================
echo.

git --version >nul 2>&1
if errorlevel 1 (
  echo ERROR: Git no esta disponible.
  pause
  exit /b 1
)

python --version
if errorlevel 1 (
  echo ERROR: Python no esta disponible.
  pause
  exit /b 1
)

echo.
echo [1/7] Actualizando resultados y datos base de Eliteserien...
python update_data.py
if errorlevel 1 goto :ERROR_UPDATE

echo.
echo [2/7] Actualizando estadisticas FotMob...
python update_fotmob_stats_v2.py
if errorlevel 1 goto :ERROR_UPDATE

echo.
echo [3/7] Reconstruyendo estadisticas de equipos...
python build_team_stats.py
if errorlevel 1 goto :ERROR_UPDATE

echo.
echo [4/7] Reentrenando modelo de goles...
python goals_model_v4_champion.py
if errorlevel 1 goto :ERROR_UPDATE

echo.
echo [5/7] Reentrenando modelos de corners y tarjetas...
python corners_model_v1.py
if errorlevel 1 goto :ERROR_UPDATE

python cards_model_v2.py
if errorlevel 1 goto :ERROR_UPDATE

echo.
echo [6/7] Preparando cambios para Git...
git add data/eliteserien.db
git add data/NOR_raw.csv
git add data/eliteserien_2023_onwards.csv
git add data/football_data_NOR_latest.csv
git add models/

git diff --cached --quiet
if not errorlevel 1 (
  echo.
  echo No hay cambios nuevos que subir.
  echo Eliteserien ya estaba actualizada.
  pause
  exit /b 0
)

echo.
echo [7/7] Creando commit y subiendo a GitHub...
git commit -m "Update Eliteserien data and models"
if errorlevel 1 (
  echo.
  echo ERROR: No se pudo crear el commit.
  pause
  exit /b 1
)

git push origin main
if errorlevel 1 (
  echo.
  echo ERROR: El push a GitHub no se completo.
  echo Copia el error y pasamelo.
  pause
  exit /b 1
)

echo.
echo ============================================================
echo OK - ELITESERIEN ACTUALIZADA Y SUBIDA A GITHUB
echo Streamlit Cloud deberia redeplegar automaticamente.
echo ============================================================
echo.
pause
exit /b 0

:ERROR_UPDATE
echo.
echo ============================================================
echo ERROR - LA ACTUALIZACION DE ELITESERIEN SE HA DETENIDO
echo NO se hara commit ni push para evitar subir datos incompletos.
echo Copia el error que aparezca arriba y pasamelo.
echo ============================================================
echo.
pause
exit /b 1
