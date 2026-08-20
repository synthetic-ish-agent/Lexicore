@echo off
setlocal
if exist .venv\Scripts\python.exe (set PY=.venv\Scripts\python.exe) else (set PY=python)
echo === chroma_db ===
%PY% inspect_db.py --db .\chroma_db
if exist .\chroma_data (
  echo.
  echo === chroma_data ===
  %PY% inspect_db.py --db .\chroma_data
)
pause
