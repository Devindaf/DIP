@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3 -m venv .venv
  if errorlevel 1 (
    echo Install Python 3.10-3.13 from python.org with the Python launcher enabled.
    pause
    exit /b 1
  )
)
.venv\Scripts\python.exe -c "import cv2,streamlit,numpy,pandas,PIL; assert cv2.__version__.startswith('4.')" >nul 2>&1
if errorlevel 1 (
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Installation failed. Check your Internet connection and Python version.
    pause
    exit /b 1
  )
)
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
pause
