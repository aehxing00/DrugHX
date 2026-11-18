@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title DrugHX 依赖安装器

echo ============================================================
echo   DrugHX 依赖安装器
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未检测到 Python。
    echo.
    echo 请安装 Python 3.10 或更高版本，安装时务必勾选
    echo   Add python.exe to PATH
    echo.
    echo 即将打开 Python 官方下载页面。
    start "" https://www.python.org/downloads/windows/
    echo.
    echo 安装完成后请关闭本窗口，然后重新运行 DrugHX。
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] 正在创建独立运行环境 ...
    python -m venv .venv
    if errorlevel 1 goto :failed
)

echo [2/4] 正在安装依赖包，首次安装可能需要几分钟 ...
".venv\Scripts\python.exe" -m pip install --upgrade pip --disable-pip-version-check
".venv\Scripts\python.exe" -m pip install -r "installer\requirements.txt" --disable-pip-version-check
if errorlevel 1 goto :failed

echo [3/4] 正在校验安装 ...
".venv\Scripts\python.exe" -c "import fastapi, uvicorn, rdkit, meeko, scipy, matplotlib, numpy, pypdf, PIL, networkx, gemmi"
if errorlevel 1 goto :failed

echo [4/4] 正在默认安装对接引擎（smina / QuickVina 2）...
".venv\Scripts\python.exe" -m dockforge.engines

echo.
echo [完成] 依赖安装成功。关闭本窗口后重新运行 DrugHX 即可。
pause
exit /b 0

:failed
echo.
echo ============================================================
echo [失败] 依赖安装未完成。
echo.
echo 常见原因与解决办法：
echo   1. 未安装 Python 3.10+ 或未加入 PATH：
echo      https://www.python.org/downloads/windows/
echo   2. 缺少 Microsoft Visual C++ 运行库：
echo      https://aka.ms/vs/17/release/vc_redist.x64.exe
echo   3. 网络无法访问 PyPI，请检查代理或稍后重试。
echo.
echo 正在打开需要安装的页面，请按提示安装。
echo 安装完成后请关闭本窗口，再重新运行 DrugHX。
echo ============================================================
start "" https://www.python.org/downloads/windows/
start "" https://aka.ms/vs/17/release/vc_redist.x64.exe
pause
exit /b 1
