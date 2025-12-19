@echo off
REM AIAS Launcher - AI Assistant System
REM Sets up environment and launches AIAS

REM Use D: drive for HuggingFace cache (to avoid filling C: drive)
set HF_HOME=D:\HuggingFaceCache
set TRANSFORMERS_CACHE=D:\HuggingFaceCache

REM Activate virtual environment and run
call "%~dp0venv\Scripts\activate.bat"
python main.py %*
