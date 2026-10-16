@echo off
chcp 65001 >nul
rem Запуск через Python из ComfyUI (там уже есть PyTorch + CUDA + spandrel)
set "PY=%LOCALAPPDATA%\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\pythonw.exe"
if not exist "%PY%" (
  echo Не найден Python ComfyUI: %PY%
  pause
  exit /b 1
)
start "" "%PY%" "%~dp0upscaler.py"
