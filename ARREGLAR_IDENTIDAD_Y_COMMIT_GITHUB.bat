@echo off
setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo FOOTBALL EDGE PRO - ARREGLAR IDENTIDAD GIT Y HACER COMMIT
echo ============================================================
echo.

git --version
if errorlevel 1 (
  echo ERROR: Git no esta disponible.
  pause
  exit /b 1
)

echo.
echo [1/5] Limpiando variables de entorno Git que pueden pisar la configuracion...
set GIT_AUTHOR_NAME=
set GIT_AUTHOR_EMAIL=
set GIT_AUTHOR_DATE=
set GIT_COMMITTER_NAME=
set GIT_COMMITTER_EMAIL=
set GIT_COMMITTER_DATE=

echo.
echo [2/5] Configurando identidad SOLO para este repositorio...
git config --local user.name "juanmacuadrabernal-dotcom"
git config --local user.email "juanmacuadrabernal@gmail.com"

echo.
echo [3/5] Verificando identidad real que usara Git...
echo Nombre local:
git config --local --get user.name
echo Email local:
git config --local --get user.email
echo.
echo Identidad calculada por Git:
git var GIT_AUTHOR_IDENT

if errorlevel 1 (
  echo.
  echo ERROR: Git sigue sin reconocer la identidad.
  echo Copia exactamente lo que aparezca arriba y pasamelo.
  pause
  exit /b 1
)

echo.
echo [4/5] Preparando archivos...
git add .

echo.
echo [5/5] Creando primer commit...
git commit -m "Football Edge Pro v14 - premium mobile dashboard"

if errorlevel 1 (
  echo.
  echo ERROR: No se pudo crear el commit.
  echo Copia exactamente lo que aparezca arriba y pasamelo.
  pause
  exit /b 1
)

echo.
echo ============================================================
echo OK - PRIMER COMMIT CREADO
echo ============================================================
git log -1 --oneline
echo.
echo Siguiente paso: crear el repo vacio "football-edge-pro" en GitHub
echo y ejecutar SUBIR_A_GITHUB.bat
echo.
pause
