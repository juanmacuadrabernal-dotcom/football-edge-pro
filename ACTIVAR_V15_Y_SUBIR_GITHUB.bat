@echo off
setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo FOOTBALL EDGE PRO - ACTIVAR V15 Y SUBIR A GITHUB
echo ============================================================
echo.

if not exist "app_v15_analista_visual.py" (
  echo ERROR: Falta app_v15_analista_visual.py
  pause
  exit /b 1
)

python -m py_compile app_v15_analista_visual.py
if errorlevel 1 (
  echo ERROR: La V15 no pasa la comprobacion de sintaxis.
  pause
  exit /b 1
)

if exist "streamlit_app.py" (
  if not exist "streamlit_app_V14_BACKUP.py" (
    copy /Y "streamlit_app.py" "streamlit_app_V14_BACKUP.py" >nul
  )
)

copy /Y "app_v15_analista_visual.py" "streamlit_app.py" >nul

echo V15 activada como streamlit_app.py
echo Preparando Git...
git add streamlit_app.py app_v15_analista_visual.py ABRIR_FOOTBALL_EDGE_PRO_V15.bat

git diff --cached --quiet
if not errorlevel 1 (
  echo No hay cambios nuevos para subir.
  pause
  exit /b 0
)

git commit -m "Football Edge Pro v15 - analyst visual workflow"
if errorlevel 1 (
  echo ERROR: No se pudo crear el commit.
  pause
  exit /b 1
)

git push origin main
if errorlevel 1 (
  echo ERROR: No se pudo subir a GitHub.
  pause
  exit /b 1
)

echo.
echo ============================================================
echo OK - V15 SUBIDA A GITHUB
echo Streamlit Cloud deberia redeplegar automaticamente.
echo ============================================================
echo.
pause
