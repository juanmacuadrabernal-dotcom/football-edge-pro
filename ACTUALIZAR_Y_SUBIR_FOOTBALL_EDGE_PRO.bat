@echo off
setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo FOOTBALL EDGE PRO - ACTUALIZAR + SUBIR A GITHUB
echo ============================================================
echo.

git --version >nul 2>&1
if errorlevel 1 (
  echo ERROR: Git no esta disponible.
  pause
  exit /b 1
)

echo [1/5] Actualizando LaLiga con el updater incremental...
python update_laliga_all_v3_incremental_fotmob.py

if errorlevel 1 (
  echo.
  echo ERROR: La actualizacion de LaLiga ha fallado.
  echo NO se hara commit ni push.
  pause
  exit /b 1
)

echo.
echo [2/5] Preparando cambios para Git...
git add -A

echo.
echo [3/5] Comprobando si hay cambios...
git diff --cached --quiet
if not errorlevel 1 (
  echo No hay cambios nuevos que subir.
  echo La app ya estaba actualizada.
  pause
  exit /b 0
)

echo.
echo [4/5] Creando commit...
git commit -m "Update LaLiga data and models"

if errorlevel 1 (
  echo.
  echo ERROR: No se pudo crear el commit.
  pause
  exit /b 1
)

echo.
echo [5/5] Subiendo a GitHub...
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
echo OK - DATOS Y MODELOS ACTUALIZADOS EN GITHUB
echo Streamlit Cloud deberia redeplegar automaticamente.
echo ============================================================
echo.
pause
