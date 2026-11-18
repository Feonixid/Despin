@echo off
rem Runs the control experiments on your GPU (RTX 5060 Ti, 16 GB is plenty for the 1.5B model).
rem It first does a 2-problem smoke test, then the full 60-problem controls, then 5 sampling seeds.
rem Results are appended to results\controls.jsonl and finished runs are skipped, so you can stop
rem (Ctrl+C) and start this file again any time. Expect several hours in total (leave it overnight).
rem Uses the CUDA build of torch that you already have; it does not reinstall torch.
cd /d "%~dp0"

python -m pip install -e . --no-deps
python -m pip install transformers accelerate numpy scipy sympy
if errorlevel 1 goto fail

echo === Smoke test (2 problems, DESPIN only) ===
python scripts\run_controls.py --configs despin --limit 2 --out results\smoke.jsonl
if errorlevel 1 goto fail

echo === Controls: greedy, despin, rollback-only, random-direction, budget-matched, closure-only ===
python scripts\run_controls.py --configs greedy despin c1_rollback_only c2_random_direction c3_despin_budget c5_closure_only --out results\controls.jsonl
if errorlevel 1 goto fail

echo === Sampling baseline, 5 seeds ===
python scripts\run_controls.py --configs sampling --seeds 5 --out results\controls.jsonl
if errorlevel 1 goto fail

echo.
echo ALL CONTROLS FINISHED.  Results: %~dp0results\controls.jsonl
echo Send me that file (or its summary) and I will add the control results to the paper.
pause
exit /b 0

:fail
echo.
echo FAILED - read the message above, then tell me what it says.
pause
exit /b 1
