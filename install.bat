@echo off

echo Starting installation for ImaGym...

REM Define the directory of the script
SET SCRIPT_DIR=%~dp0

REM Change to the directory where the script is located
cd /d %SCRIPT_DIR%

REM Check if Python is installed
python3 --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python 3 could not be found. Please install Python 3 and rerun this script.
    exit /b 1
)

echo Python 3 detected...

REM Install virtualenv if it's not already installed
python3 -m pip show virtualenv >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Installing virtualenv...
    python3 -m pip install virtualenv
)

REM Create a Python virtual environment
echo Creating a Python virtual environment...
python3 -m virtualenv venv

REM Activate the virtual environment
echo Activating the virtual environment...
CALL venv\Scripts\activate.bat


REM Change into the app/ directory
cd app/

REM Check for and install requirements.txt
IF EXIST requirements.txt (
    echo Installing required Python packages...
    python3 -m pip install -r requirements.txt
) ELSE (
    echo requirements.txt not found. Ensure you have the requirements.txt file in the same directory as this script.
    exit /b 1
)

echo Installation complete. You can now run the application.

REM Additional instructions to run the application can be included here
REM For example:
REM echo To run the application, use: python3 app.py

