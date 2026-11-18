@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title DrugHX 启动器

if not exist ".venv\Scripts\python.exe" goto :install
".venv\Scripts\python.exe" -c "import fastapi, uvicorn, rdkit, meeko, scipy, matplotlib, numpy, pypdf, PIL, networkx, gemmi" >nul 2>nul
if errorlevel 1 goto :install

".venv\Scripts\python.exe" -m dockforge.launcher %*
if errorlevel 1 (
    echo.
    echo [DrugHX] 启动失败，请查看上方错误信息，或把 data\logs\dockforge.log 发给开发者。
    pause
)
goto :eof

:install
echo [DrugHX] 首次运行或依赖缺失，正在启动依赖安装器 ...
call "%~dp0安装依赖.bat"
if errorlevel 1 goto :eof
".venv\Scripts\python.exe" -m dockforge.launcher %*
if errorlevel 1 (
    echo.
    echo [DrugHX] 启动失败，请查看上方错误信息，或把 data\logs\dockforge.log 发给开发者。
    pause
)
goto :eof
