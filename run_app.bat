@echo off

echo Running ImaGym...

REM Define the directory of the script
SET SCRIPT_DIR=%~dp0

REM Change to the directory where the script is located
cd /d %SCRIPT_DIR%

REM Check and install requirements
IF NOT EXIST venv (
    echo Setting up the environment and installing requirements...
    CALL install.bat
) ELSE (
    echo Environment already set up.
    CALL venv\Scripts\activate.bat
)

REM Change into the app/ directory
cd app/

REM Run the Python script
echo Running ui.py...
python3 ui.py