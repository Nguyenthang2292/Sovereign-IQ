@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
.venv\Scripts\python.exe scripts\run_adaptive_trend_parity_harness.py --fixtures-dir tests\parity_fixtures --strict --verbose
