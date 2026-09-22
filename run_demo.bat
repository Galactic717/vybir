@echo off
REM vybir quick demo: real typed decisions on examples\ (CPU/CUDA auto-detect).
set PYTHONPATH=%~dp0
python -m vybir predict --state-file "%~dp0examples\state.json" --questions "%~dp0examples\questions.json" %*
