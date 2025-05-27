@echo off
echo === Enemy Agent CUDA Training ===
echo.

echo Step 1: Verifying CUDA extensions...
python verify_cuda_extensions.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: CUDA extensions verification failed.
    exit /b 1
)

echo.
echo Step 2: Training enemy agent with custom CUDA...
python train_enemy_cuda.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Training failed.
    exit /b 1
)

echo.
echo === Training completed successfully! ===
echo The enemy agent was trained using custom C++/CUDA modules on your NVIDIA GPU.
echo.