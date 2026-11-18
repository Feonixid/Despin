@echo off
rem Regenerates every number in the paper from the raw logs, runs the tests, and builds paper\main.pdf.
rem Needs: Python on PATH, and a LaTeX install (MiKTeX or TeX Live) for the last step.
cd /d "%~dp0"

echo === 1/4  Installing the small analysis dependencies ===
python -m pip install numpy scipy sympy pytest
if errorlevel 1 goto fail

echo === 2/4  Regenerating all paper numbers, tables and figure data from data\logged_run ===
python scripts\make_paper_assets.py
if errorlevel 1 goto fail

echo === 3/4  Running tests ===
python -m pytest -q
if errorlevel 1 goto fail

echo === 4/4  Building the paper ===
cd paper
where latexmk >nul 2>nul
if %errorlevel%==0 (
  latexmk -pdf -interaction=nonstopmode main.tex
) else (
  pdflatex -interaction=nonstopmode main.tex
  pdflatex -interaction=nonstopmode main.tex
  pdflatex -interaction=nonstopmode main.tex
)
cd ..
echo.
echo DONE.  Paper: %~dp0paper\main.pdf
pause
exit /b 0

:fail
echo.
echo FAILED - read the message above, then tell me what it says.
pause
exit /b 1
