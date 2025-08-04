@echo off
echo UAV Strategic Deconfliction System
echo FlytBase Robotics Assignment 2025
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ and try again
    pause
    exit /b 1
)

echo Installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Running the UAV Deconfliction System...
echo.

python main.py

if errorlevel 1 (
    echo.
    echo ERROR: Program execution failed
    pause
    exit /b 1
)

echo.
echo ====================================
echo Execution completed successfully!
echo Check the output/ directory for generated files
echo ====================================
pause