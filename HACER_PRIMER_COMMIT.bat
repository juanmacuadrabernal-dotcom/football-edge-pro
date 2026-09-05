@echo off
cd /d "%~dp0"

echo.
echo ============================================================
echo FOOTBALL EDGE PRO - PRIMER COMMIT
echo ============================================================
echo.

git --version
if errorlevel 1 (
  echo ERROR: Git no esta disponible.
  pause
  exit /b 1
)

for /f "delims=" %%A in ('git config user.name') do set GITNAME=%%A
for /f "delims=" %%A in ('git config user.email') do set GITEMAIL=%%A

if "%GITNAME%"=="" (
  echo Git necesita tu nombre para firmar el commit.
  set /p GITNAME=Escribe tu nombre o nick de GitHub: 
  git config user.name "%GITNAME%"
)

if "%GITEMAIL%"=="" (
  echo.
  echo Git necesita un email para firmar el commit.
  echo Puedes usar el email de GitHub o uno noreply.
  set /p GITEMAIL=Escribe tu email: 
  git config user.email "%GITEMAIL%"
)

echo.
echo Nombre Git: 
git config user.name
echo Email Git:
git config user.email

echo.
echo Revisando staging...
git status --short

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
echo Siguiente paso: crear el repositorio vacio en GitHub.
echo Nombre recomendado: football-edge-pro
echo.
pause
