@echo off
echo AI Platform Trainer - Command Examples
echo =====================================

echo Choose an option:
echo.
echo 1. Regular game (with GUI)
echo 2. Headless training (no display, 2x speed)
echo 3. Headless training (no display, 10x speed)
echo 4. Start directly in play mode (skip menu)
echo 5. Start directly in training mode (skip menu)
echo 6. Headless training with batch logging
echo 7. Exit
echo.

set /p option="Enter option (1-7): "

if "%option%"=="1" (
    echo Running regular game...
    python -m ai_platform_trainer.main
) else if "%option%"=="2" (
    echo Running headless training at 2x speed...
    python -m ai_platform_trainer.main --headless --mode train --training-speed 2.0
) else if "%option%"=="3" (
    echo Running headless training at 10x speed...
    python -m ai_platform_trainer.main --headless --mode train --training-speed 10.0
) else if "%option%"=="4" (
    echo Starting directly in play mode...
    python -m ai_platform_trainer.main --mode play
) else if "%option%"=="5" (
    echo Starting directly in training mode...
    python -m ai_platform_trainer.main --mode train
) else if "%option%"=="6" (
    echo Running headless training with batch logging...
    python -m ai_platform_trainer.main --headless --mode train --batch-logging --training-speed 5.0
) else if "%option%"=="7" (
    echo Exiting...
    exit /b
) else (
    echo Invalid option. Please try again.
    pause
    exit /b
)

echo.
echo Done.
pause
