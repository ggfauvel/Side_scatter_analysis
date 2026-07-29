@echo off
REM ============================================================
REM  Spectro Multi-Fibres - lanceur Windows (double-cliquez-moi)
REM  Premier lancement : cree l'environnement et installe les
REM  dependances (quelques minutes, connexion internet requise).
REM  Lancements suivants : demarrage direct.
REM ============================================================
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
    echo Python n'est pas installe ou pas dans le PATH.
    echo Installez Python 3.10+ depuis https://www.python.org/downloads/
    echo IMPORTANT : cochez "Add Python to PATH" pendant l'installation.
    pause
    exit /b 1
)
if not exist ".venv" (
    echo Premiere installation : creation de l'environnement...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)
python run.py
pause
