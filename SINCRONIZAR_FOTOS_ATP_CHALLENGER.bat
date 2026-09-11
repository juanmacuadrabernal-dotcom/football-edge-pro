@echo off
setlocal
cd /d "%~dp0"

title TENNIS EDGE PRO - FOTOS ATP + CHALLENGER

echo ============================================================================
echo TENNIS EDGE PRO - SINCRONIZAR FOTOS OFICIALES ATP + CHALLENGER
echo ============================================================================
echo.
echo Se revisaran todos los jugadores unicos de tennis_edge.db.
echo El proceso es incremental: si lo cierras, al ejecutarlo otra vez continua.
echo.
echo Las imagenes NO se guardan dentro de SQLite.
echo Se guardan ATP ID + URL oficial de ATP Tour.
echo.
echo NO CIERRES esta ventana si quieres completar toda la primera sincronizacion.
echo ============================================================================
echo.

python sync_player_photos.py

echo.
echo ============================================================================
echo SINCRONIZACION TERMINADA
echo ============================================================================
echo.
python comprobar_fotos.py

echo.
pause
endlocal
