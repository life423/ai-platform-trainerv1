@echo off
echo Training Missile Model...
python -m ai_platform_trainer.ai_model.train --model missile
echo.
echo Training Enemy Model...
python -m ai_platform_trainer.ai_model.train --model enemy
echo.
echo All models trained successfully!
