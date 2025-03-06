@echo off
echo AI Platform Trainer
echo =================
echo Available options:
echo   --headless         Run in headless mode without display
echo   --mode train/play  Start directly in train or play mode
echo   --batch-logging    Enable batch logging (save data at end of session)
echo   --training-speed X Run training at X times normal speed (e.g., 2.0)
echo.

python -m ai_platform_trainer.main %*
