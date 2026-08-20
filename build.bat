@echo off
cd /d "%~dp0"
rmdir /s /q dist build saddle_llm.egg-info 2>nul
python -m build --wheel
echo.
echo Done: saddle_llm
dir dist\*.whl
