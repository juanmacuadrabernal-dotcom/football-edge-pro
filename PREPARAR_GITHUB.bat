@echo off
cd /d "%~dp0"
echo.
echo ================================================
echo FOOTBALL EDGE PRO - PREPARAR GITHUB
echo ================================================
echo.
git --version
if errorlevel 1 (
  echo.
  echo ERROR: Git no esta instalado o no esta en PATH.
  echo Instala Git for Windows y vuelve a ejecutar.
  pause
  exit /b 1
)

if not exist ".git" (
  git init
)

git branch -M main
git add .
git status
echo.
echo Si arriba esta todo correcto, el siguiente paso sera conectar el repositorio GitHub.
echo.
pause
