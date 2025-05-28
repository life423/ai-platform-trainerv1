@echo off
REM Build script for CUDA extension
REM Run this from Developer Command Prompt for VS 2022

echo ============================================================
echo CUDA Extension Build Script
echo ============================================================
echo.

REM Check if we're in the Developer Command Prompt
cl >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Visual Studio C++ compiler not found!
    echo Please run this script from "Developer Command Prompt for VS 2022"
    echo.
    echo To fix this:
    echo 1. Close this window
    echo 2. Open "Developer Command Prompt for VS 2022" from Start Menu
    echo 3. Navigate to: cd "%~dp0"
    echo 4. Run: build_cuda.bat
    echo.
    pause
    exit /b 1
)

echo ✓ Visual Studio C++ compiler found

REM Check CUDA
nvcc --version >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: CUDA compiler (nvcc) not found!
    echo Please ensure CUDA is installed and in PATH
    pause
    exit /b 1
)

echo ✓ CUDA compiler found

REM Check Python environment
python --version >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found!
    echo Please activate your virtual environment first
    pause
    exit /b 1
)

echo ✓ Python found

REM Activate virtual environment if not already active
if not defined VIRTUAL_ENV (
    echo Activating virtual environment...
    call "..\..\..\.venv\Scripts\activate.bat"
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Could not activate virtual environment
        pause
        exit /b 1
    )
)

echo ✓ Virtual environment active

REM Check required packages
python -c "import pybind11" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing pybind11...
    pip install pybind11
)

echo ✓ pybind11 available

echo.
echo ============================================================
echo Building CUDA Extension
echo ============================================================
echo.

REM Clean previous build
if exist "build" (
    echo Cleaning previous build...
    rmdir /s /q "build"
)

if exist "gpu_environment.*.pyd" (
    echo Removing previous extension...
    del "gpu_environment.*.pyd"
)

REM Build the extension
echo Building extension...
python setup.py build_ext --inplace

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo BUILD SUCCESS!
    echo ============================================================
    echo.
    
    REM Test the extension
    echo Testing extension...
    python -c "import gpu_environment; print('✓ GPU extension imported successfully!')"
    
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo Extension is ready to use!
        echo You can now run your GPU RL pipeline tests.
    ) else (
        echo.
        echo Extension built but import failed. Check dependencies.
    )
    
) else (
    echo.
    echo ============================================================
    echo BUILD FAILED!
    echo ============================================================
    echo.
    echo Check the error messages above for details.
    echo Common issues:
    echo - Missing Visual Studio components
    echo - CUDA version compatibility
    echo - Missing dependencies
)

echo.
pause
