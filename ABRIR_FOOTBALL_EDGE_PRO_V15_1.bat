@echo off
cd /d "%~dp0"
start "" cmd /c "timeout /t 3 /nobreak >nul & start "" http://localhost:8501"
python -m streamlit run app_v15_1_visual_fiel.py --server.port 8501
pause
