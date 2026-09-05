@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo.
echo ============================================================
echo FOOTBALL EDGE PRO - PRIMER COMMIT CORREGIDO
echo ============================================================
echo.

git --version
if errorlevel 1 (
  echo ERROR: Git no esta disponible.
  pause
  exit /b 1
)

echo Configurando identidad Git...
git config --global user.name "juanmacuadrabernal-dotcom"
git config --global user.email "juanmacuadrabernal@gmail.com"

echo.
echo Nombre Git:
git config --global user.name
echo Email Git:
git config --global user.email

echo.
echo Preparando archivos...
git add .

echo.
echo Creando primer commit...
git commit -m "Football Edge Pro v14 - premium mobile dashboard"

if errorlevel 1 (
  echo.
  echo ERROR: No se pudo crear el commit.
  echo Copia lo que aparezca arriba y pasamelo.
  pause
  exit /b 1
)

echo.
echo ============================================================
echo OK - PRIMER COMMIT CREADO
echo ============================================================
git log -1 --oneline
echo.
echo Ya puedes crear el repositorio VACIO en GitHub:
echo football-edge-pro
echo.
pause
