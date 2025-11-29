@echo off
echo ==========================================
echo 🚀 Starting Full MLOps Local Environment
echo ==========================================

REM -------- CONFIG --------
set PROJECT_PATH=D:\New-folder\mlops-churn-pipeline
set VENV_PATH=%PROJECT_PATH%\env
set API_PATH=src\serving\api:app
set UI_PATH=ui\app.py

echo.
echo 🔧 Activating Virtual Environment...
call %VENV_PATH%\Scripts\activate

echo.
echo ▶️ Starting FastAPI backend at http://127.0.0.1:8000 ...
start "FastAPI Server" cmd /k "cd %PROJECT_PATH% && uvicorn %API_PATH% --reload"

echo.
echo ▶️ Starting Streamlit UI at http://localhost:8501 ...
start "Streamlit UI" cmd /k "cd %PROJECT_PATH% && streamlit run %UI_PATH%"

echo.
echo 🎉 Everything is running!
echo Close this window if needed.
pause
