@echo off
cd /d "%~dp0"

echo.
echo ============================================================
echo FOOTBALL EDGE PRO - SUBIR A GITHUB
echo ============================================================
echo.

echo Antes de seguir:
echo 1. Crea en GitHub un repositorio VACIO llamado football-edge-pro
echo 2. NO marques README, .gitignore ni License
echo 3. Copia la URL HTTPS del repositorio
echo.
set /p REPOURL=Pega aqui la URL HTTPS de GitHub: 

if "%REPOURL%"=="" (
  echo ERROR: No has pegado ninguna URL.
  pause
  exit /b 1
)

echo.
echo Configurando origin...
git remote remove origin >nul 2>&1
git remote add origin "%REPOURL%"

echo.
echo Subiendo rama main...
git branch -M main
git push -u origin main

if errorlevel 1 (
  echo.
  echo ERROR: El push no se completo.
  echo Si GitHub abre el navegador para iniciar sesion, completalo y vuelve a ejecutar.
  pause
  exit /b 1
)

echo.
echo ============================================================
echo OK - FOOTBALL EDGE PRO SUBIDO A GITHUB
echo ============================================================
echo.
git remote -v
echo.
pause
